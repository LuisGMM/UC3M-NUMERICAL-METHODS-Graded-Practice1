
import sys
import warnings
from typing import Callable

import numpy as np



def newton(err:float, f:'Callable[float]' = None, f_dev:'Callable[float]' = None,
    composite:'Callable[Callable, float, float, float]' = None,  c:float = 0, x0:float = 0, h_err:float = 1e-4) -> float:
    """Newton's method to find roots of a function.
    
    If no `f` is given but `f_dev` and `composite` are, it will compute the roots of the integral of `f_dev` with integration constant c.
    If `f_dev` is not given, it will be computed from `f` with the mathematical definition of a derivative.

    Args:
        err (float): Desired error of the method.
        f_dev (Callable[float], optional): Analytical function to find its roots. Its input is the point to be evaluated in. Defaults to None.
        f_dev (Callable[float], optional): Analytical derivative of the function. Its input is the point to be evaluated in. Defaults to None.
        composite (Callable[Callable, float, float, float], optional): Integration method to compute the integral of `f_dev` and find its roots. 
            It should be `composite_trapezoid` or `composite_simpson` methods. Defaults to None.
        c (float, optional): Integration constant of the integral of f_dev. Defaults to 0.
        x0 (float, optional): Initial guess of the root. 
            Note that an inadequate first guess could lead to undesired outputs such as no roots or undesired roots.
            Defaults to 0.
        h_err (float, optional): Finite approximation of 0 to use in the calculation of `f_dev` by its mathematical definition. Defaults to 1e-4.

    Returns:
        float|None: Root of the function or None if the algorithm reaches its recursion limit.
    """    
    def dev(x:float, f:'Callable[float]' = f, h_err:float = h_err)-> float:
        return ( f(x+h_err) - f(x) ) / h_err 

    if (f or composite) and f_dev:
        if f and composite:
            warnings.warn('`f`, `f_dev` and `composite` args detected. Only `f` and `f_dev` will be used for sake of precision.') 
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

        elif composite:
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - (composite(f_dev, x0, iter_dict[iter_idx], 100_000) + c) / f_dev(iter_dict[iter_idx])

        else:
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

    elif f and f_dev == None:
        warnings.warn(f'`f_dev` was not given. It will be computed using the derivative definition with `h`={h_err} .') 
        iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / dev(x=iter_dict[iter_idx], f=f)
    
    else:
        raise type('InadequateArgsCombination', (Exception,), {})('Cannot compute Newton s method with the combination of arguments given. Check the valid combinations.')

    iter, iter_dict = 0, {0:x0}
    limit = sys.getrecursionlimit()

    while True:
        if iter + 10 >= limit:
            warnings.warn(f'Iteration limit ({limit}) reached without finding any root. Try with other initial guess or changing the recursion limit. Maybe there are no roots.')
            return 
        
        iter_dict[iter+1] = iteration(iter, iter_dict)
        
        if abs(iter_dict[iter+1] - iter_dict[iter]) < err:
            return iter_dict[iter+1]
        
        iter += 1


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

def euler_explicit_midpoint(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float)-> np.ndarray:
    """Computes the explicit (forward) midpoint Euler method to solve ODEs.

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
    t_ = np.arange(t0, t0+t, h) 
    N = len(t_)

    u = np.zeros_like(t_)
    u_previous = y0 - h * f(y0, t_[0])
    u[0] = y0
 
    for i in range(N-1):
        if i == 0:
            u[i+1] = u_previous +2 * h * f(u[i], t_[i])    
        else:
            u[i+1] = u[i-1] + 2 *h * f(u[i], t_[i])
    
    return u

def euler_implicit(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float, *args, **kwargs)-> np.ndarray:
    """Computes the implicit (backward) Euler method to solve ODEs.

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
    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0
        
    for i in range(N-1):        

        g = lambda y: u[i] + u[i+1] + h*f(y, t_[i+1])
        u[i+1] = newton(*args, f=g, x0=u[i], **kwargs)

    return u


if __name__ == '__main__':
    pass
    # f_t_y = lambda y,t: - (3* t**2 * y + y**2) / (2* t**3 + 3* t*y)
    # h_vec = [0.0001, 0.001, 0.01, 0.1]

    # for hi in h_vec:
    #     print(f"Euler implicit wit h={hi} yields {euler_implicit(f_t_y, -2, 1, 2, hi, 1e-1)}")    
        