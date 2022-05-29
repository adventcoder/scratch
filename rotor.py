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

    def project(self, v):
        '''The projection of another vector onto this vector.'''
        return self * (self.dot(v) / self.dot(self))

    def rotate(self, v, angle):
        '''Rotate another vector about this vector.'''
        a, b = math.cos(angle), math.sin(angle)
        return a * v + b * self.unit().cross(v) + (1 - a) * self.project(v)

    def unit(self):
        '''self == abs(self) * self.unit()'''
        try:
            return self / abs(self)
        except ZeroDivisionError:
            # We can write as 0 times any unit vector, so just pick one.
            # This vector will be used for the return value of log(-1) and sqrt(-1).
            return Vector3D(1, 0, 0)

    def __abs__(self):
        return math.sqrt(self.abs_squared())

    def abs_squared(self):
        return self.dot(self)

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
        if is_scalar(b):
            return Vector3D(a.x / b, a.y / b, a.z / b)
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

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

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
        Take the x, y, z axes as pointing right, forward, up respectively.
        The returned rotor then rotates a vector upwards around the x axis.
        '''
        return cls(math.cos(angle), Vector3D(math.sin(angle), 0, 0))

    # Rotor3D.roll(B).matrix() == [[ cos(B), 0, sin(B)],
    #                              [      0, 1,      0],
    #                              [-sin(B), 0, cos(B)]]
    @classmethod
    def roll(cls, angle):
        '''
        Take the x, y, z axes as pointing right, forward, up respectively.
        The returned rotor then rotates a vector to the right around the y axis.
        '''
        return cls(math.cos(angle), Vector3D(0, math.sin(angle), 0))

    # Rotor3D.yaw(C).matrix() == [[cos(C), -sin(C), 0],
    #                             [sin(C),  cos(C), 0],
    #                             [     0,       0, 1]]
    @classmethod
    def yaw(cls, angle):
        '''
        Take the x, y, z axes as pointing right, forward, up respectively.
        The returned rotor then rotates a vector to the left around the z axis.
        '''
        return cls(math.cos(angle), Vector3D(0, 0, math.sin(angle)))

    @classmethod
    def axis_angle(cls, axis, angle):
        '''
        Construct a rotor that rotates a vector around a general axis of rotation.
        The direction of the rotation will be according to the left/right hand rule depending on the coordinate system.
        If the axis is not a unit vector then the vector to be rotated is also scaled by the magnitude of the axis.
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
        return cls(radius * math.cos(angle), radius * math.sin(angle) * normal.unit())

    @classmethod
    def euler(cls, A, B, C):
        # return cls.pitch(A) * cls.roll(B) * cls.yaw(C)
        Ax, Ay = math.cos(A), math.sin(A)
        Bx, By = math.cos(B), math.sin(B)
        Cx, Cy = math.cos(C), math.sin(C)
        a =  Ax*Bx*Cx + Ay*By*Cy
        b = -Ax*By*Cy + Ay*Bx*Cx
        c =  Ax*By*Cx + Ay*Bx*Cy
        d =  Ax*Bx*Cy - Ay*By*Cx
        return cls(a, Vector3D(b, c, d))

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
        return Rotor3D(a*a - n.dot(n), 2*a*n)

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
        # ¬r v r = (a^2-n.n) v + 2 a nxv + 2 n.v n
        #        = ((a^2-b^2-c^2-d^2) [1 0 0] + 2 a [ 0 -d  c] + 2 [bb bc bd]) v
        #                             [0 1 0]       [ d  0 -b]     [cb cc cd]
        #                             [0 0 1]       [-c  b  0]     [db dc dd]
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        a2 = a*a - b*b - c*c - d*d
        b2 = 2*a*b
        c2 = 2*a*c
        d2 = 2*a*d
        return [[ a2 + 2*b*b, -d2 + 2*b*c,  c2 + 2*b*d],
                [ d2 + 2*c*b,  a2 + 2*c*c, -b2 + 2*c*d],
                [-c2 + 2*d*b,  b2 + 2*d*c,  a2 + 2*d*d]]

    def apply(self, v):
        #
        # ¬sqrt(r) v sqrt(r) = a v + nxv + (|r|-a) n.v/n.n n
        #
        # Doesn't compose nicely since in general: sqrt(r1 r2) != sqrt(r1) sqrt(r2).
        #
        # Note that sqrt(-1) will be chosen such that ¬sqrt(-1) v sqrt(-1) = -v.
        # Or in otherwords sqrt(-1) = v/|v| i j k.
        #
        if isinstance(v, Vector3D):
            a, n = self.sym, self.skew
            if n.dot(n) == 0:
                return a*v
            else:
                return a*v + n.cross(v) + (abs(self) - a) * n.project(v)
        elif isinstance(v, Rotor3D):
            return Rotor3D(v.sym * abs(self), self.apply(v.skew))
        elif is_scalar(v):
            return v * abs(self)
        else:
            return NotImplemented

    def matrix(self):
        a, n = self.sym, self.skew
        if n.dot(n) == 0:
            return [[a, 0, 0],
                    [0, a, 0],
                    [0, 0, a]]
        else:
            b, c, d = n.x, n.y, n.z
            m = (abs(self) - a) / n.dot(n)
            return [[ a + m*b*b, -d + m*b*c,  c + m*b*d],
                    [ d + m*c*b,  a + m*c*c, -b + m*c*d],
                    [-c + m*d*b,  b + m*d*c,  a + m*d*d]]

    def euler_angles(self):
        # q = a + b j k + c k i + d i j
        # q = |q| (cos(A)+sin(A) j k)(cos(B)+sin(B) k i)(cos(C)+sin(C) i j)
        #
        # a = |q| ( cos(A)cos(B)cos(C)+sin(A)sin(B)sin(C))
        # b = |q| (-cos(A)sin(B)sin(C)+sin(A)cos(B)cos(C))
        # c = |q| ( cos(A)sin(B)cos(C)+sin(A)cos(B)sin(C))
        # d = |q| ( cos(A)cos(B)sin(C)-sin(A)sin(B)cos(C))
        #
        # a + c = |q| cos(A-C)(cos(B)+sin(B))
        # a - c = |q| cos(A+C)(cos(B)-sin(B))
        # b + d = |q| sin(A+C)(cos(B)-sin(B))
        # b - d = |q| sin(A-C)(cos(B)+sin(B))
        #
        # A + C = atan2(b+d,a-c)
        # A - C = atan2(b-d,a+c)
        #   2 B = asin(2(a c-b d)/|q|^2)
        #
        #---
        #
        # Using below properties euler angles can always be reduced to be in the range:
        # * -pi   <= A <= pi
        # * -pi/4 <= B <= pi/4
        # * -pi/2 <= C <= pi/2
        #
        # Full angle:
        # R(A+2 pi,B,C) = R(A,B+2 pi,C) = R(A,B,C+2 pi) = R(A,B,C)
        #
        # Half angle:
        # R(A,B+pi,C) = R(A+pi,B,C) = R(A,B,C+pi)
        #
        # Quarter angle:
        # R(A+pi/2,B,C+pi/2) = (cos(A+pi/2)+sin(A+pi/2) j k) (cos(B)+sin(B) k i) (cos(C+pi/2)+sin(C+pi/2)i j)
        #                    = (-sin(A)+cos(A) j k) (cos(B)+sin(B) k i) (-sin(C)+cos(C) i j)
        #                    = (cos(A)+sin(A) j k) j k (cos(B)+sin(B) k i) i j (cos(C)+sin(C) i j)
        #                    = (cos(A)+sin(A) j k) (sin(B)+cos(B) k i) (cos(C)+sin(C) i j)
        #                    = (cos(A)+sin(A) j k) (cos(pi/2-B)+sin(pi/2-B) k i) (cos(C)+sin(C) i j)
        #                    = R(A,pi/2-B,C)
        #
        a, b, c, d = self.sym, self.skew.x, self.skew.y, self.skew.z
        r2 = a*a + b*b + c*c + d*d
        x = math.atan2(b + d, a - c)
        y = math.atan2(b - d, a + c)
        B = math.asin(2*(a*c-b*d) / r2) / 2
        return (x + y) / 2, B, (x - y) / 2

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
        # Divison by a rotor isn't uniquely defined.
        # Multiply by the reciprocal on either the left or right instead.
        if is_scalar(b):
            return Rotor3D(a.sym / b, a.skew / b)
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
            return a.sym == b and a.skew == 0
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

if __name__ == '__main__':
    # Example usage:
    pos = Vector3D(0, 0, 0)
    rot = Rotor3D(1) # no rotation

    # Turn up 30 degrees:
    rot *= Rotor3D.pitch(math.radians(30))

    # Turn left 45 degrees:
    rot *= Rotor3D.yaw(math.radians(45))

    # Move in the direction faced by applying the rotor to the "forward" vector.
    vel = rot.apply(Vector3D(0, 10, 0))
    pos += vel
    print("now at", repr(pos))

    # Find rotation from one vector onto another as euler angles:
    vec1 = Vector3D(0, 1, 0)
    vec2 = Vector3D(0, -1, 1)
    print("euler angles:", list(map(math.degrees, (vec1*vec2).euler_angles())))
    print("matrix:", (vec1.unit()*vec2.unit()).matrix())
