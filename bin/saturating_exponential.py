import logging
import itertools
from collections import Iterable

import numpy as np

from mle import mle, log_likelihood, APPROX_ZERO

logging.basicConfig(level=logging.DEBUG)

A_GUESSES =  [i/10.0 for i in xrange(1, 10)]
B_GUESSES = [i/10.0 for i in xrange(1, 10)]
T_GUESSES = [1000, 500, 250, 100, 50, 10]
# BOUNDS = {'A': (0.0, 1.0), 'B': (0.0, 1.0), 'T': (0.0, None)}
BOUNDS = {'A': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'B': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'T': (0.0+APPROX_ZERO, None)}
CONSTRAINTS = [] # [{'type': 'ineq', 'fun': lambda theta: theta[0] - theta[1]}] # A > B

def get_guesses(A, B, T):
    if A is None and B is None and T is None:
        guesses = [A_GUESSES, B_GUESSES, T_GUESSES]
    elif A is None and B is not None and T is None:
        guesses = [A_GUESSES, T_GUESSES]
    elif A is not None and B is not None and T is None:
        guesses = [T_GUESSES]
    else:
        msg = "SAT-EXP incorrectly defined."
        logging.error(msg)
        raise Exception(msg)
    return list(itertools.product(*guesses)) # cartesian product

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

def log_likelihood_fcn(data, (A, B, T)):
    if A is None and B is None and T is None:
        return lambda theta: -log_likelihood(data, saturating_exp, (theta[0], theta[1], theta[2]))
    elif A is None and B is not None and T is None:
        return lambda theta: -log_likelihood(data, saturating_exp, (theta[0], B, theta[1]))
    elif A is not None and B is not None and T is None:
        return lambda theta: -log_likelihood(data, saturating_exp, (A, B, theta[0]))
    else:
        msg = "SAT-EXP incorrectly defined."
        logging.error(msg)
        raise Exception(msg)

def fit(data, (A, B, T), quick, guesses=None, method='TNC'):
    """
    (A, B, T) are numerics
        the model parameters
        if None, they will be fit
    """
    thetas = (A, B, T)
    if guesses is None:
        guesses = get_guesses(A, B, T)
    bounds = [BOUNDS[key] for key, val in zip(['A', 'B', 'T'], [A, B, T]) if val is None]
    constraints = CONSTRAINTS
    return mle(data, log_likelihood_fcn(data, thetas), guesses, bounds, constraints, quick, method)
