
import sys
import warnings
from typing import Callable

import numpy as np



def newton(composite:'Callable[Callable, float, float, float]', f_dev:'Callable[float]', c:float, err:float, x0:float = 0) -> float:
    """Newton's method to find roots of the integral of f_dev with integration constant c.

    This method should be used when only the analitycal derivative f_dev of the function is known.
    composite arg should be composite_trapezoid or composite_simpson callable methods.

    Args:
        composite (Callable[Callable, float, float, float]): _description_
        f_dev (Callable[float]): Analytical derivative of the function. It input is the point to be evaluated in.
        c (float): Integration constant of the integral of f_dev. 
        err (float): Desired error of the method.
        x0 (float, optional): Initial guess of the root. 
            Note that an inadequate first guess could lead to undesired outputs such as no roots or undesired roots.
            Defaults to 0.

    Returns:
        float: Root of the function.
    """    
    iter_value = {0:x0}
    iter=0
    limit = sys.getrecursionlimit()

    while True:
        
        if iter + 10 >= limit:
            warnings.warn('Iteration limit reached without finding any root. Try with other initial guess. Maybe there are no roots.')
            return 
        
        iter_value[iter+1] = iter_value[iter] - (composite(f_dev, x0, iter_value[iter], 100000) + c) / f_dev(iter_value[iter])
        iter += 1

        if abs(iter_value[iter] - iter_value[iter-1]) < err:
            return iter_value[iter]
        

def composite_trapezoid(f_:'Callable[float]', a:float, b:float, n:float)-> float:
    """Computes the analitical solution of the integral of f from a to b 
    following the composite trapezoidal rule. 

    Args:
        f_ (Callable[float]): Function to be integrated  
        a (float): Lower bound of hte interval.
        b (float): Upper bound of the interval.
        n (float): The number of parts the interval is divided into.

    Returns:
        float: Numerical solution of the integral.
    """    
    x = np.linspace(a, b, n + 1)
    f = f_(x)
    h = (b - a) / (n)

    return h/2 * sum(f[:n] + f[1:n+1])


def composite_simpson(f_:'Callable[float]', a:float, b:float, n:float)-> float:
    """Computes the analitical solution of the integral of f from a to b 
    following the composite Simpson's 1/3 rule. 

    Args:
        f_ (Callable[float]): Function to be integrated  
        a (float): Lower bound of hte interval.
        b (float): Upper bound of the interval.
        n (float): The number of parts the interval is divided into.

    Returns:
        float: Numerical solution of the integral.
    """    
    x = np.linspace(a, b, n+1)
    f = f_(x)
    h = (b - a) / (n)
    integral_value = (h/3) * ( f[0] + 2*sum(f[2:n-1:2]) + 4*sum(f[1:n:2]) + f[n] )
    
    return integral_value


def euler_explicit(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float)-> np.ndarray:
    """Computes the explicit (forward) Euler method to solve ODEs.

    Args:
        f (Callable[float, float]): Function depending on y and t in that order.
            Equivalent to f(y,t).
        y0 (float): Initial value of the answer.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, t-h, t].
    """
    t_ = np.arange(t0, t0+t, h) #TODO: Seems wrong to me 
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0
 
    for i in range(N-1):
        u[i+1] = u[i] + h * f(u[i], t_[i])
    
    return u

def euler_implicit(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float)-> np.ndarray:
    pass