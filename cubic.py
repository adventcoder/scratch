import math

def cbrt(x):
    '''Return the principal cube root of x.'''
    if x == 0:
        return x
    return math.copysign(math.exp(math.log(math.fabs(x)) / 3), x)

def cbrt2(p, q = 0):
    '''
    Return all solutions to: t^3 = p + q t.
    When q is 0, the same as cbrt(p), except returns all three solutions.
    '''
    #
    # t^3 = p + q t
    # (u^3+v^3) + 3 u v (u+v) = p + q (u+v)
    #
    # u^3 + v^3 = p
    # 3 u v = q
    #
    # u^3 = p/2 + sqrt((p/2)^2 - (q/3)^3) = x + i y
    # v^3 = p/2 - sqrt((p/2)^2 - (q/3)^3) = x - i y
    #
    # x = p/2
    # y = sqrt((q/3)^3 - (p/2)^2)
    #
    x = p / 2
    x2 = x*x
    r2 = q*q*q / 27
    if x2 < r2:
        # u^3 = x + i y = r e^( i a)
        # v^3 = x - i y = r e^(-i a)
        # u = cbrt(r) e^( i a/3)
        # v = cbrt(r) e^(-i a/3)
        r = math.sqrt(r2)
        a = math.acos(x / r)
        r_prime = cbrt(r)
        x_prime = r_prime * math.cos(a / 3)
        y_prime = r_prime * math.sin(a / 3)
    else:
        yj = math.sqrt(x2 - r2)
        u = cbrt(x + yj)
        v = cbrt(x - yj)
        x_prime = (u + v) / 2
        y_prime = (u - v) / 2j
    # tk = e^(i (2 pi)/3 k) u + e^(-i (2 pi)/3 k) v
    #
    # x' = (u+v)/2
    # y' = (u-v)/(2 i)
    #
    # t0 = u + v = 2 x'
    # t1 = -1/2 (u + v) + i sqrt(3)/2 (u - v) = -x' - sqrt(3) y'
    # t2 = -1/2 (u + v) - i sqrt(3)/2 (u - v) = -x' + sqrt(3) y'
    #
    t0 = 2 * x_prime
    t1 = -x_prime - math.sqrt(3)*y_prime
    t2 = -x_prime + math.sqrt(3)*y_prime
    return (t0, t1, t2)

def cubic_roots(a, b, c, d):
    '''Return all solutions to: a x^3 + b x^2 + c x + d = 0'''
    B = b / a
    C = c / a
    D = d / a
    # x^3 + B x^2 + C x + D = 0
    # t^3 + (3 u+B) t^2 + (3 u^2+2 B u+C) t + (u^3+B u^2+C u+D) = 0, x=t+u
    u = -B/3
    p = D + u*(C + u*(B + u))
    q = C + u*(2*B + 3*u)
    t0, t1, t2 = cbrt2(-p, -q)
    return (t0 + u, t1 + u, t2 + u)

def cubic_coefficients(a, b, c, m = 1):
    '''Returns the coefficients of the cubic polynomial with roots a,b,c and leading coefficient m.'''
    e1 = a + b + c
    e2 = a*b + a*c + b*c
    e3 = a*b*c
    return (m, -m*e1, m*e2, -m*e3)
