from math import isnan, sqrt, erf
from collections import Iterable
import logging

import numpy as np

logging.basicConfig(level=logging.DEBUG)
is_nan = lambda x: abs(x) == float('inf') or isnan(x)
APPROX_ZERO = 0.0001

def keep_solution(theta, bnds, ymin):
    """
    theta is return value of scipy.optimize.minimize,
        where theta['x'] is list [t1, ...] of solution values
    bnds is list [(lb1, rb1), ...] of bounds for each ti in theta['x']
    ymin is previously-found minimum solution

    returns True iff theta is a success, lower than ymin, and has solutions not near its bounds
    """
    if not theta['success']:
        return False
    if theta['fun'] >= ymin:
        return False
    close_enough = lambda x, b: abs(x-b) < APPROX_ZERO*10
    at_left_bound = lambda x, lb: close_enough(x, lb) if lb else False
    at_right_bound = lambda x, rb: close_enough(x, rb) if rb else False
    at_bounds = lambda x, (lb, rb): at_left_bound(x, lb) or at_right_bound(x, rb)
    return not any([at_bounds(th, bnd) for th, bnd in zip(theta['x'], bnds)])

def log_likelihood(arr, fcn, thetas, try_modifying=False):
    """
    arr is array of [[x0, y0], [x1, y1], ...]
        where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi
    try_modifying corrects each 

    calculates the sum of the log-likelihood of arr
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    fcn_x = lambda x: fcn(x, *thetas)
    fcn_x_adjust = lambda x: fcn(x, *thetas)*0.99 + 0.005 if try_modifying else fcn(x, *thetas)
    likelihood = lambda row: fcn_x_adjust(row[0]) if row[1] else 1-fcn_x_adjust(row[0])
    log_likeli = lambda row: np.log(likelihood(row))
    val = sum(map(log_likeli, arr))
    if is_nan(val) and not try_modifying:
        val = log_likelihood(arr, fcn, thetas, try_modifying=True)
    return val

def saturating_exp(x, A, B, T):
    """
    A is the end asymptote value
    B is the start asymptote value
    T is the time s.t. value is 63.2%  of (A-B)
    """
    if T == 0:
        # returns either A or list of As depending on how whether x is numeric or array
        return np.ones(len(x))*A if isinstance(x, Iterable) else A
    return A - (A-B)*np.exp(-x*1000.0/T)

def twin_limb(x, x0, S0, p):
    """
    p is the slope at which the value increases
    x0 is the time at which the value plateaus
    S0 is the value after the value plateaus

    twin-limb function [Burr, Santoto (2001)]
        S(x) =
                {
                    S0 * (x/x0)^p | x < x0
                    S0             | x >= x0
                }
    """
    return S0 if x >= x0 else S0*pow(x/x0, p)

def drift_diffusion((t, C), k):
    """
    t, C are independent variables
    k relates the drift rate to C

    From Selen et al. (2012):
        dX = kCdt + dW
            X = choice {0, 1}
            C = coherence
            t = duration
            W = standard Brownian motion with unit variance over one second
        This yields (according to the paper):
            f(X) = N(kCt, t) = gaussian with mean kCt, variance t
        =>
            p(C, t) = P(X == 1 | C, t) = 1 - (f(X) <= 0) = 1 - F(0)
            n.b. F(0) = 0.5 + 0.5*erf(-kCt / sqrt(2*t))]
        =>
            p(C, t) = 1 - F(0) = 0.5 - 0.5*erf(-kCt / sqrt(2*t))
    """
    return 0.5 - 0.5*erf(-k*C*t / sqrt(2*t))
