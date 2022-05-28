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
        Satisfies: self == abs(self) * self.unit()
        '''
        r = abs(self)
        if r == 0:
            # We can write as 0 times any unit vector.
            # We can't know what the correct unit vector is so just pick one.
            # This vector will be used for the return value of log(-1) and sqrt(-1).
            return Vector3D(0, 0, 1)
        else:
            return self / r

    def rotate(self, axis, angle):
        return math.cos(angle) * self + math.sin(angle) * axis.cross(self) + (1 - math.cos(angle)) * axis.dot(self) * axis.unit()

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
        '''
        self.sym = float(sym)
        self.skew = skew

    # Rotor3D.pitch(A).matrix() == [[1,       0,       0],
    #                               [0,  cos(A), -sin(A)],
    #                               [0,  sin(A),  cos(A)]]
    @classmethod
    def pitch(cls, angle):
        '''
        Take the x, y, z axes as pointing right, forward, up respectively (right handed).
        This method then returns a rotor that rotates a vector upwards around the x axis.
        '''
        return cls(math.cos(angle), Vector3D(math.sin(angle), 0, 0))

    # Rotor3D.roll(B).matrix() == [[ cos(B), 0, sin(B)],
    #                              [      0, 1,      0],
    #                              [-sin(B), 0, cos(B)]]
    @classmethod
    def roll(cls, angle):
        '''
        Take the x, y, z axes as pointing right, forward, up respectively (right handed).
        This method then returns a rotor that rotates a vector right around the y axis.
        '''
        return cls(math.cos(angle), Vector3D(0, math.sin(angle), 0))

    # Rotor3D.yaw(C).matrix() == [[cos(C), -sin(C), 0],
    #                             [sin(C),  cos(C), 0],
    #                             [     0,       0, 1]]
    @classmethod
    def yaw(cls, angle):
        '''
        Take the x, y, z axes as pointing right, forward, up respectively (right handed).
        This method then returns a rotor that rotates a vector to the left around the z axis.
        '''
        return cls(math.cos(angle), Vector3D(0, 0, math.sin(angle)))

    @classmethod
    def axis_angle(cls, axis, angle):
        '''
        Construct a rotor that rotates a vector around a general axis of rotation.
        The direction of the rotation will be according to the left/right hand rule depending on the coordinate system.
        If the axis is not a unit vector then the vector to be rotated is also scaled by the length of the axis.
        Satisfies: self = Rotor3D.axis_angle(abs(self) * self.axis(), self.angle())
        '''
        return cls(math.cos(angle) * abs(axis), math.sin(angle) * axis)

    @classmethod
    def polar(cls, radius, normal):
        '''
        Construct a rotor that rotates a vector around the given normal vector and that scales by the given radius.
        The angle of rotation corresponds to the magnitude of the normal vector.
        Satisfies: self = Rotor3D.polar(abs(self), self.arg())
        '''
        angle = abs(normal)
        return cls(radius * math.cos(angle), radius * sin(angle) * normal.unit())

    @classmethod
    def euler(cls, A, B, C):
        return cls.pitch(A) * cls.roll(B) * cls.yaw(C)

    def __abs__(self):
        return math.sqrt(self.abs_squared())

    def abs_squared(self):
        return self.sym*self.sym + self.skew.dot(self.skew)

    def arg(self):
        return self.angle() * self.axis()

    def axis(self):
        '''
        Returns the axis of rotation as a unit vector.
        An arbitrary unit vector is returned when the angle is 0 or pi.
        '''
        return self.skew.unit()

    def angle(self):
        '''Returns the angle of rotation (between 0-pi)'''
        return math.atan2(abs(self.skew), self.sym)

    def exp(self):
        #
        # e^(v i j k) = sum(n=0->inf, (v i j k)^n/n!)
        #             = sum(n=0->inf, (v i j k)^(2 n)/(2 n)!) + sum(n=0->inf, (v i j k)^(2 n+1)/(2 n+1)!)
        #             = sum(n=0->inf, (-|v|^2)^n/(2 n)!) + sum(n=0->inf, (-|v|^2)^n/(2 n+1)!) v i j k
        #             = sum(n=0->inf, (-1)^n |v|^(2 n)/(2 n)!) + sum(n=0->inf, (-1)^n |v|^(2 n+1)/(2 n+1)!) v/|v| i j k
        #             = cos|v| + sin|v| v/|v| i j k
        #
        return Rotor3D.polar(math.exp(self.sym), self.skew)

    def log(self):
        return Rotor3D(math.log(self.abs_squared()) / 2, self.arg())

    def squared(self):
        a, n = self.sym, self.skew
        return Rotor3D(a*a - n.dot(n), 2 * a * n)

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
        # sqrt(q) = sqrt((|q|+a)/2) + sqrt((|q|-a)/2) v/|v| i j k
        #
        a = (abs(self) + self.sym) / 2
        # Don't want to deal with edge cases so just do the two square roots.
        return Rotor3D(math.sqrt(a), math.sqrt(a - self.sym) * self.skew.unit())

    def apply_squared(self, v):
        #
        # Rotation formula:
        #
        # ¬r v r = (a-n i j k) v (a+n i j k)
        #        = a^2 v + a (v n-n v) i j k + n v n
        #        = (a^2 - n.n) v + 2 a nxv + 2 n n.v
        #
        # Normal vector is unchanged (except for scaling):
        #
        # ¬r n r = |r|^2 n
        #
        # Composes with multiplication:
        #
        # (r1*r2).apply_squared(v) = ¬(r1 r2) v (r1 r2)
        #                          = ¬r2 (¬r1 v r1) r2
        #                          = r2.apply_squared(r1.apply_squared(v))
        #
        if isinstance(v, Vector3D):
            a, n = self.sym, self.skew
            return (a*a-n.dot(n))*v + 2*a*n.cross(v) + 2*n.dot(v)*n
        elif isinstance(v, Rotor3D):
            # ¬r (a+n i j k) r = (|r|^2 a) + (¬r n r) i j k
            return Rotor3D(v.sym * self.abs_squared(), self.apply_squared(v.skew))
        elif is_scalar(v):
            return v * self.abs_squared()
        else:
            return NotImplemented

    def matrix_squared(self):
        #
        # r v ¬r = (a^2-n.n) v + 2 a nxv + 2 n.v n
        #        = ((a^2-b^2-c^2-d^2) [1 0 0] + 2 a [ 0 -d  c] + 2 [bb bc bd]) v
        #                             [0 1 0]       [ d  0 -b]     [cb cc cd]
        #                             [0 0 1]       [-c  b  0]     [db dc dd]
        #        = [a^2+b^2-c^2-d^2     -2ad+2bc          2ac+2bd    ] v
        #          [    2ad+2cb      a^2-b^2+c^2-d^2     -2ab+2cd    ]
        #          [   -2ac+2db          2ab+2dc      a^2-b^2-c^2+d^2]
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        a2 = a*a - b*b - c*c - d*d
        return [
            [  a2 + 2*b*b, 2*(-a*d+b*c), 2*( a*c+b*d)],
            [2*( a*d+c*b),   a2 + 2*c*c, 2*(-a*b+c*d)],
            [2*(-a*c+d*b), 2*( a*b+d*c),   a2 + 2*d*d]
        ]

    def apply(self, v):
        #
        # ¬sqrt(r) v sqrt(r) = a v + nxv + (|r|-a) (n/|n|).v (n/|n|)
        #
        # Doesn't compose nicely since in general: sqrt(r1 r2) != sqrt(r1) sqrt(r2).
        #
        if isinstance(v, Vector3D):
            a, n = self.sym, self.skew
            m = math.sqrt(a*a + n.dot(n)) - a
            u = n.unit()
            return a*v + n.cross(v) + m * u.dot(v) * u
        elif isinstance(v, Rotor3D):
            return Rotor3D(abs(self), self.apply(v.skew))
        elif isscalar():
            return v * abs(self)
        else:
            return NotImplemented

    def matrix(self):
        a, n = self.sym, self.skew
        m = math.sqrt(a*a + n.dot(n)) - a
        u = n.unit()
        return [
            [   a + m*u.x*u.x, -n.z + m*u.x*u.y,  n.y + m*u.x*u.z],
            [ n.z + m*u.y*u.x,    a + m*u.y*u.y, -n.x + m*u.y*u.z],
            [-n.y + m*u.z*u.x,  n.x + m*u.z*u.y,    a + m*u.z*u.z]
        ]

    def euler_angles(self):
        '''
        Returns the euler angles A, B, C that correspond to applying this rotor to a vector.

        - A: rotation around the x axis, -pi<=A<=pi
        - B: rotation around the y axis, -pi/2<=B<=pi/2
        - C: rotation around the z axis, -pi<=C<=pi
        '''
        # q = a + b j k + c k i + d i j
        # q = |q| (cos(A)+sin(A) j k)(cos(B)+sin(B) k i)(cos(C)+sin(C) i j)
        #
        # matrix(q)^2 = |q|^2 matrix(cos(C)+sin(C) i j)^2 matrix(cos(B)+sin(B) k i)^2 matrix(cos(A)+sin(A) j k)^2
        #
        # [...] = |q|^2 [cos(2C) -sin(2C) 0] [ cos(2B) 0 sin(2B)] [1    0        0   ] = |q|^2 [ cos(2B)cos(2C)         ...             ...      ]
        # [...]         [sin(2C)  cos(2C) 0] [    0    1    0   ] [0 cos(2A) -sin(2A)]         [ cos(2B)sin(2C)         ...             ...      ]
        # [...]         [   0        0    1] [-sin(2B) 0 cos(2B)] [0 sin(2A)  cos(2A)]         [-sin(2B)          cos(2B)sin(2A)   cos(2B)cos(2A)]
        #
        # a^2 + b^2 - c^2 - d^2 = |q|^2 cos(2B)cos(2C)
        #          2(a d + c b) = |q|^2 cos(2B)sin(2C)
        #          2(a c - b d) = |q|^2 sin(2B)
        #          2(a b + c d) = |q|^2 cos(2B)sin(2A)
        # a^2 - b^2 - c^2 + d^2 = |q|^2 cos(2B)cos(2A)
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        r2 = a*a + b*b + c*c + d*d
        Cx = r2 - 2*(c*c + d*d)
        Cy = 2*(a*d + c*b)
        By = 2*(a*c - b*d)
        Ay = 2*(a*b + c*d)
        Ax = r2 - 2*(c*c + b*b)
        return math.atan2(Ay, Ax), math.asin(By / r2), math.atan2(Cy, Cx)

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
            return Rotor3D(b + a.sym, a.skew)
        else:
            return NotImplemented

    def __sub__(a, b):
        if isinstance(b, Rotor3D):
            return Rotor3D(a.sym - b.sym, a.skew - b.skew)
        elif is_scalar(b):
            return Rotor3D(a.sym - b, a.skew)
        else:
            return NotImplemented

    def __rsub__(a, b):
        if is_scalar(b):
            return Rotor3D(b - a.sym, -a.skew)
        else:
            return NotImplemented

    def __mul__(a, b):
        if isinstance(b, Rotor3D):
            # (a1+v1 i j k) (a2+v2 i j k) = (a1 a2 - v1.v2) + (a1 v2 + v1 a2 - v1xv2) i j k
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
    M = list(zip(normalise(a), normalise(b), *map(normalise, normals(a, b))))
    # Then to rotate in the plane ab do a change of basis to M, rotate about xy, then change back.
    # Rab(t) = Mab Rxy(t) Mab^-1
    R = identity_matrix(len(M))
    R[0][0] =  math.cos(angle)
    R[0][1] = -math.sin(angle)
    R[1][0] = -R[0][1]
    R[1][1] =  R[0][0]
    return matrix_mulitply(M, matrix_mulitply(R, inverse_matrix(M)))

def matrix_mulitply(A, B):
    return [[dot(a, b) for b in zip(*B)] for a in A]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def normalise(xs):
    n = math.sqrt(dot(xs, xs))
    return list(x / n for x in xs)

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

if __name__ == '__main__':

    a = [1, 2, 3]
    b = [1, -1, 1]

    axis = Vector3D(*a).cross(Vector3D(*b)).unit()
    angle = math.pi / 3

    for row in rotation_matrix(a, b, angle):
        print(row)
    print()
    for row in Rotor3D.axis_angle(axis, angle / 2).matrix_squared():
        print(row)
