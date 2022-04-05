

import numpy as np

from utilmethods import (composite_simpson, composite_trapezoid,
                         euler_explicit, euler_implicit, newton)


def question1():

    def f_dev(x):
        return 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)


    ERR = 1e-5
    p0 = 0.5
    x0 = 0
    C = -0.45
    print(newton(composite_simpson, f_dev, C, ERR, x0))
    print(newton(composite_trapezoid, f_dev, C, ERR, x0))



def question2():
    
    def f(y, t):
        return np.sqrt(1 + t**3)
    
    y0 = 0
    t0 = 0
    t = 5

    h = 1/64

    while h <= 1:
        print(euler_explicit(f, y0, t0, t, h))
        print(euler_implicit(f, y0, t0, t, h))
        h *= 2