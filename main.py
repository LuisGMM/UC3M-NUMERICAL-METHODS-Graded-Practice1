

import numpy as np

from utilmethods import (composite_simpson, composite_trapezoid,
                         euler_explicit, euler_implicit, newton)


def question1():

    def f_dev(x):
        return 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

    ERR = 1e-5
    p0 = 0.5
    x0 = p0
    C = -0.45
    print(newton(composite=composite_simpson, f_dev=f_dev, c=C, err=ERR, x0=x0))
    print(newton(composite=composite_trapezoid, f_dev=f_dev, c=C, err=ERR, x0=x0))


def question2():

    def f(y, t):
        return np.sqrt(1 + t**3)

    y0 = 0
    t0 = 0
    t = 5

    h = 1/64

    while h <= 1:
        print(euler_explicit(f, y0, t0, t, h))
        print(euler_implicit(f, y0, t0, t, h, 1e-5))
        h *= 2

question2()