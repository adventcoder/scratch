import math

def is_scalar(x):
    return isinstance(x, (float, int))

def format_expr(vals, units):
    strs = []
    for val, unit in zip(vals, units):
        if val == 0:
            continue
        if len(strs) > 0:
            if val < 0:
                strs.append('-')
                val = -val
            else:
                strs.append('+')
        if abs(val) == 1 and unit is not None:
            strs.append('-' + unit if val < 0 else unit)
        else:
            strs.append(str(val))
            if unit is not None:
                strs.append(unit)
    if not strs:
        return '0'
    return ' '.join(strs)

class Vector3D:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def dot(a, b):
        return a.x*b.x + a.y*b.y + a.z*b.z

    def cross(a, b):
        x = a.y*b.z - a.z*b.y
        y = a.x*b.z - a.z*b.x
        z = a.x*b.y - a.y*b.x
        return Vector3D(x, -y, z)

    def wedge(a, b):
        return Rotor3D(0, a.cross(b))

    def unit(self):
        '''
        Satisfies: self = abs(self) * self.unit()
        '''
        r = abs(self)
        if r == 0:
            # We can write as 0 times any unit vector.
            # We can't know what the correct unit vector is so we'll just pick one.
            # This vector will be used for the return value of log(-1) and sqrt(-1).
            return Vector3D(0, 0, 1)
        else:
            return self / r

    def reciprocal(self):
        return self / self.dot(self)

    def __abs__(self):
        return math.sqrt(self.dot(self))

    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)

    def __add__(a, b):
        if isinstance(b, Vector3D):
            return Vector3D(a.x + b.x, a.y + b.y, a.z + b.z)
        else:
            return NotImplemented

    def __sub__(a, b):
        if isinstance(b, Vector3D):
            return Vector3D(a.x - b.x, a.y - b.y, a.z - b.z)
        else:
            return NotImplemented

    def __mul__(a, b):
        if isinstance(b, Vector3D):
            # u v = (u1 i + u2 j + u3 k) (v1 i + v2 j + v3 k)
            #     = (u1 v1 + u2 v2 + u3 v3) + (u1 v2-u2 v1) i j - (u1 v3-u3 v1) k i + (u2 v3-u3 v2) j k
            #     = u.v + uxv i j k
            return Rotor3D(a.dot(b), a.cross(b))
        elif is_scalar(b):
            return Vector3D(a.x * b, a.y * b, a.z * b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if is_scalar(b):
            return Vector3D(b * a.x, b * a.y, b * a.z)
        else:
            return NotImplemented

    def __truediv__(a, b):
        if isinstance(b, Vector3D):
            return a * b.reciprocal()
        elif is_scalar(b):
            return Vector3D(a.x / b, a.y / b, a.z / b)
        else:
            return NotImplemented

    def __rtruediv__(a, b):
        if is_scalar(b):
            return b * a.reciprocal()
        else:
            return NotImplemented

    def __eq__(a, b):
        if isinstance(b, Vector3D):
            return a.x == b.x and a.y == b.y and a.z == b.z
        elif is_scalar(b):
            return b == 0 and a.x == 0 and a.y == 0 and a.z == 0
        else:
            return NotImplemented

    def __hash__(self):
        if self.x == 0 and self.y == 0 and self.z == 0:
            return hash(0)
        else:
            return hash((self.x, self.y, self.z))

    def __bool__(self):
        return self != 0

    def __str__(self):
        return format_expr([self.x, self.y, self.z], ['i', 'j', 'k'])

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.x, self.y, self.z)

class Rotor3D:
    def __init__(self, sym, skew = Vector3D(0, 0, 0)):
        '''
        Construct a rotor that represents the geometric product of two vectors.

        Fields:
        - sym: the symmetric scalar part of the product, represented here as a float.
        - skew: the anti-symmetric bivector part of the product, represented here as a Vector3D.

        Let a denote the symmetric part.
        Let v = b i + c j + d k denote the anti-symmetric part as a vector.

        Then the rotor can be written as:
        a + v i j k

        In polar form a rotor can be written as:
        r e^(n i j k) = r (cos|n| + sin|n| n/|n| i j k)

        - r is the radius of the rotor, the scaling component.
        - n is a vector normal to the plain of rotation.
        - |n|, the magnitude of the vector gives the angle to rotate through.

        If n is a unit vector then: (n i j k)^2 = -1.
        This means that there are infinitely many square roots of -1, all the unit vectors.
        '''
        self.sym = float(sym)
        self.skew = skew

    # Rotor3D.pitch(A).matrix() == [[1,        0,       0],
    #                               [0,  cos(2A), sin(2A)],
    #                               [0, -sin(2A), cos(2A)]]
    @classmethod
    def pitch(cls, angle):
        '''Construct a rotor that rotates a vector around the x (left-right) axis (yz plane).'''
        return cls(math.cos(angle), Vector3D(math.sin(angle), 0, 0))

    # Rotor3D.yaw(B).matrix() == [[cos(2B), 0, -sin(2B)],
    #                             [      0, 1,        0],
    #                             [sin(2B), 0,  cos(2B)]]
    @classmethod
    def yaw(cls, angle):
        '''Construct a rotor that rotates a vector around the y (up-down) axis (zx plane).'''
        return cls(math.cos(angle), Vector3D(0, math.sin(angle), 0))

    # Rotor3D.roll(C).matrix() == [[ cos(2C), sin(2C), 0],
    #                              [-sin(2C), cos(2C), 0],
    #                              [       0,       0, 1]]
    @classmethod
    def roll(cls, angle):
        '''Construct a rotor that rotates a vector around the z (forward-back) axis (xy plane).'''
        return cls(math.cos(angle), Vector3D(0, 0, math.sin(angle)))

    @classmethod
    def axis_angle(cls, axis, angle):
        '''
        Construct a rotor that rotates a vector around a general axis of rotation.
        If the axis is not a unit vector then the vector is also scaled by the length of the axis.
        '''
        return cls(math.cos(angle) * abs(axis), math.sin(angle) * axis)

    @classmethod
    def polar(cls, radius, normal):
        '''
        Construct a rotor that rotates a vector in the plane perpendicular to the given normal vector and that scales by the given radius.
        The angle of rotation corresponds to the magnitude of the normal vector.
        '''
        angle = abs(normal)
        return cls(radius * math.cos(angle), radius * sin(angle) * normal.unit())

    @classmethod
    def euler(cls, A, B, C):
        return cls.roll(C) * cls.yaw(B) * cls.pitch(A)

    def __abs__(self):
        return math.sqrt(self.abs_squared())

    def abs_squared(self):
        return self.sym*self.sym + self.skew.dot(self.skew)

    def arg(self):
        '''
        Returns the argument of this rotor, such that: self == Rotor3D.polar(abs(self), self.arg())
        The argument is a vector that is normal to the plane of rotation of this rotor.
        The magnitude of the vector gives the angle, between 0-pi.
        '''
        # q = a + v i j k
        #
        #           n = atan(|v|/|a|) v/|v|
        # e^(n i j k) = (|a| + v i j k)/|q|
        #
        #           n = (pi - atan(|v|/|a|)) v/|v|
        # e^(n i j k) = (-|a| + v i j k)/|q|
        #
        angle = math.atan2(abs(self.skew), self.sym)
        return angle * self.skew.unit()

    def sqrt(self):
        #
        # q = a + v i j k
        #
        # sqrt(a + v i j k) = a' + v' i j k
        #       a + v i j k = (a' + v' i j k)^2
        #                   = a'^2 - v'.v' + 2 a' v' i j k
        #
        # a = a'^2 - v'.v'
        # v = 2 a' v'
        #
        #     a = a'^2 - (v/(2 a')) . (v/(2 a'))
        # |q|^2 = (2 a'^2)^2 - 2 a (2 a'^2) + a^2
        #  a'^2 = (|q| + a)/2
        #    a' = sqrt((|q| + a)/2)
        #
        # v' = v/(2 a')
        #    = v/(2 sqrt((|q| + a)/2))
        #    = sqrt((|q| - a)/2) v/|v|
        #
        # sqrt((|q| + a)/2) sqrt((|q| - a)/2) = |v|/2
        #                    a' = (|v|/2)/sqrt((|q| - a)/2)
        #
        # sqrt(q) = sqrt((|q|+a)/2) + sqrt((|q|-a)/2) v/|v| i j k
        #
        a = (abs(self) + self.sym) / 2
        # Don't want to deal with edge cases so just do the two square roots.
        return Rotor3D(math.sqrt(a), math.sqrt(a - self.sym) * self.skew.unit())

    def apply(self, v):
        #
        # Rotation formula:
        #
        # r v ¬r = (a+n i j k) v (a-n i j k)
        #        = a^2 v - a (v n-n v) i j k + n v n
        #        = a^2 v - a (2 nxv) + (2 n.v n - n.n v)
        #        = (a^2-n.n) v + 2 n.v n - 2 a nxv
        #
        # Normal vector is unchanged (except for scaling):
        #
        # r n ¬r = |r|^2 n
        #
        # Composes with multiplication:
        #
        # (r1*r2).apply(v) = (r1 r2) v ¬(r1 r2)
        #                  = r1 (r2 v ¬r2) ¬r1
        #                  = r1.apply(r2.apply(v))
        #
        return (self.sym*self.sym - self.skew.dot(self.skew))*v + 2*self.skew.dot(v)*self.skew - 2*self.sym*self.skew.cross(v)

    def matrix(self):
        '''
        Returns the matrix that corresponds to applying this rotor to a vector (when right multiplied).
        '''
        #
        # r v ¬r = (a^2-n.n) v + 2 n.v n - 2 a nxv
        #        = ((a^2-b^2-c^2-d^2) [1 0 0] + 2 [bb bc bd] - 2 a [ 0 -d  c]) v
        #                             [0 1 0]     [cb cc cd]       [ d  0 -b]
        #                             [0 0 1]     [db dc dd]       [-c  b  0]
        #        = [a^2+b^2-c^2-d^2      2bc+2ad          2bd-2ac    ] v
        #          [    2cb-2ad      a^2-b^2+c^2-d^2      2cd+2ab    ]
        #          [    2db+2ac          2dc-2ab      a^2-b^2-c^2+d^2]
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        r = a*a - b*b - c*c - d*d
        return [
            [  r + 2*b*b, 2*(b*c+a*d), 2*(b*d-a*c)],
            [2*(c*b-a*d),   r + 2*c*c, 2*(c*d+a*b)],
            [2*(d*b+a*c), 2*(d*c-a*b),   r + 2*d*d]
        ]

    def euler_angles(self):
        '''
        Returns the euler angles A, B, C that correspond to applying this rotor to a vector.

        - A: rotation around the x axis, -pi<=A<=pi
        - B: rotation around the y axis, -pi/2<=B<=pi/2
        - C: rotation around the z axis, -pi<=C<=pi
        '''
        #
        # q = a + b j k + c k i + d i j
        # q = |q| (cos(C/2)+sin(C/2) i j)(cos(B/2)+sin(B/2) k i)(cos(A/2)+sin(A/2) j k)
        #
        # matrix(q) = |q|^2 matrix(cos(C/2)+sin(C/2) i j) matrix(cos(B/2)+sin(B/2) k i) matrix(cos(A/2)+sin(A/2) j k)
        #
        # [a^2+b^2-c^2-d^2      2bc+2ad          2bd-2ac    ] = |q|^2 [ cos(B)cos(C)        ...            ...     ]
        # [    2cb-2ad      a^2-b^2+c^2-d^2      2cd+2ab    ]         [-cos(B)sin(C)        ...            ...     ]
        # [    2db+2ac          2dc-2ab      a^2-b^2-c^2+d^2]         [ sin(B)         -sin(A)cos(B)   cos(A)cos(B)]
        #
        # By = 2(ac+db)/|q|^2 = sin(B)
        #
        # Ax = (a^2-b^2-c^2+d^2)/|q|^2 = cos(B) cos(A)
        # Ay = 2(ab-dc)/|q|^2          = cos(B) sin(A)
        #
        # Cx = (a^2+b^2-c^2-d^2)/|q|^2 = cos(B) cos(C)
        # Cy = 2(ad-cb)/|q|^2          = cos(B) sin(C)
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        r2 = a*a + b*b + c*c + d*d
        Ax = 1 - 2*(b*b+c*c)/r2
        Cx = 1 - 2*(c*c+d*d)/r2
        Ay = 2*(a*b-d*c)/r2
        By = 2*(a*c+d*b)/r2
        Cy = 2*(a*d-c*b)/r2
        return math.atan2(Ay, Ax), math.asin(By), math.atan2(Cy, Cx)

    def exp(self):
        #
        # e^(v i j k) = sum[n=0->inf, (v i j k)^n/n!)
        #             = sum(n=0->inf, (v i j k)^(2 n)/(2 n)!) + sum(n=0->inf, (v i j k)^(2 n+1)/(2 n+1)!)
        #             = sum(n=0->inf, (-|v|^2)^n/(2 n)!) + sum(n=0->inf, (-|v|^2)^n/(2 n+1)!) v i j k
        #             = sum(n=0->inf, (-1)^n |v|^(2 n)/(2 n)!) + sum(n=0->inf, (-1)^n |v|^(2 n+1)/(2 n+1)!) v/|v| i j k
        #             = cos|v| + sin|v| v/|v| i j k
        #
        return Rotor3D.polar(math.exp(self.sym), self.skew)

    def log(self):
        return Rotor3D(math.log(self.abs_squared()) / 2, self.arg())

    def conjugate(self):
        return Rotor3D(self.sym, -self.skew)

    def reciprocal(self):
        return self.conjugate() / self.abs_squared()

    def __neg__(self):
        return Rotor3D(-self.sym, -self.skew)

    def __add__(a, b):
        if isinstance(b, Rotor3D):
            return Rotor3D(a.sym + b.sym, a.skew + b.skew)
        elif is_scalar(b):
            return Rotor3D(a.sym + b, a.skew)
        else:
            return NotImplemented

    def __radd__(a, b):
        if is_scalar(b):
            return a + b
        else:
            return NotImplemented

    def __sub__(a, b):
        if isinstance(b, Rotor3D):
            return Rotor3D(a.sym + b.sym, a.skew + b.skew)
        elif is_scalar(b):
            return Rotor3D(a.sym + b, a.skew)
        else:
            return NotImplemented

    def __rsub__(a, b):
        if is_scalar(b):
            return -(a - b)
        else:
            return NotImplemented

    def __mul__(a, b):
        if isinstance(b, Rotor3D):
            # (a1+v1 i j k) (a2+v2 i j k) = a1 a2 + a1 v2 i j k + a2 v1 i j k - v1.v2 - v1 v2 i j k
            sym = a.sym*b.sym - a.skew.dot(b.skew)
            skew = a.sym*b.skew + a.skew*b.sym - a.skew.cross(b.skew)
            return Rotor3D(sym, skew)
        elif is_scalar(b):
            return Rotor3D(a.sym*b, a.skew*b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if is_scalar(b):
            return Rotor3D(b*a.sym, b*a.skew)
        else:
            return NotImplemented

    def __truediv__(a, b):
        if isinstance(b, Rotor3D):
            # NOTE: This is right division. Multiplying on the left by the inverse is generally different since rotor multiplication is not commutative.
            return a * b.reciprocal()
        elif is_scalar(b):
            return Rotor3D(a.sym / b, a.skew / b)
        else:
            return NotImplemented

    def __rtruediv__(a, b):
        if is_scalar(b):
            return b * a.reciprocal()
        else:
            return NotImplemented

    def __pow__(a, b):
        if isinstance(b, Rotor3D):
            if a == 0:
                return Rotor3D(math.pow(0, b.sym), 0)
            else:
                return (a.log() * b).exp()
        elif is_scalar(b):
            return Rotor3D.polar(math.pow(a.abs_squared(), b / 2), a.arg() * b)
        else:
            return NotImplemented

    def __rpow__(a, b):
        if is_scalar(b):
            if b == 0:
                return Rotor3D(math.pow(0, a.sym), 0)
            else:
                return (math.log(b) * a).exp()
        else:
            return NotImplemented

    def __eq__(a, b):
        if isinstance(b, Rotor3D):
            return a.sym == b.sym and a.skew == b.skew
        elif is_scalar(b):
            return a.sym == b and self.skew == 0
        else:
            return NotImplemented

    def __hash__(self):
        if self.skew == 0:
            return hash(self.sym)
        else:
            return hash((self.sym, self.skew))

    def __bool__(self):
        return self != 0

    def __str__(self):
        return format_expr([self.sym, self.skew.x, self.skew.y, self.skew.z], [None, 'j k', 'k i', 'i j'])

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.sym, self.skew)

#-------------------------------------------------------------------------------

def rotation_matrix(a, b, angle):
    '''
    Construct a rotation matrix that rotates by angle in the plane ab.
    '''
    # Form basis matrix Mab with a and b in place of the x and y axes.
    M = list(zip(a, b, *normals(a, b)))
    # Then to rotate in the plane ab do a change of basis to M, rotate about xy, then change back.
    # Rab(t) = Mab Rxy(t) Mab^-1
    R = identity_matrix(len(M))
    R[0][0] = math.cos(angle)
    R[0][1] = math.sin(angle)
    R[1][0] = -R[0][1]
    R[1][1] =  R[0][0]
    return dot(dot(M, zip(*R)), zip(*inverse_matrix(M)))

def dot(rows, cols):
    return [[sum(x * y for x, y in zip(row, col)) for col in cols] for row in rows]

def identity_matrix(n):
    I = [[0] * n for i in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def inverse_matrix(M):
    N = list(map(list, M))
    invert_matrix(N)
    return N

def invert_matrix(M):
    if not all(len(x) == len(M) for x in M):
        raise ValueError('matrix not square')
    for n in range(len(M)):
        if M[n][n] == 0:
            for i in range(n + 1, len(M)):
                if M[i][n] != 0:
                    M[n], M[i] = M[i], M[n]
                    break
            else:
                raise ZeroDivisionError('singular matrix')
        if M[n][n] != 1:
            x = 1 / M[n][n]
            M[n][n] = 1
            for j in range(len(M)):
                M[n][j] *= x
        for i in range(len(M)):
            if i != n and M[i][n] != 0:
                x = M[i][n]
                M[i][n] = 0
                for j in range(len(M)):
                    M[i][j] -= x * M[n][j]

def normals(a, b):
    '''
    Given two n dimensional vectors, find n-2 normal vectors.
    In 3 dimensions this is equivalent to the cross product.
    '''
    # The vector pair a, b form a hyperplane in n dimensions.
    # For another vector x on the hyperplane:
    # a.x = 0
    # b.x = 0
    # This is a linear system of equations that can be solved for x.
    # All vectors in the resulting vector space will be normal to a and b.
    #
    # x0 =  sum(k=2->n-1, (a1 bk-ak b1) tk)
    # x1 = -sum(k=2->n-1, (a0 bk-ak b0) tk)
    # xk = (a0 b1-a1 b0) tk, k=2->n-1
    #
    n = min(len(a), len(b))
    xs = []
    for i in range(2, n):
        x = [0] * n
        x[0] =  (a[1]*b[i] - a[i]*b[1])
        x[1] = -(a[0]*b[i] - a[i]*b[0])
        x[i] =  (a[0]*b[1] - a[1]*b[0])
        xs.append(x)
    return xs
