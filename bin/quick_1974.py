import logging
import itertools
from math import exp

from mle import mle, log_likelihood, APPROX_ZERO

logging.basicConfig(level=logging.DEBUG)

BOUNDS = {'A': (0.0+APPROX_ZERO, 100.0-APPROX_ZERO), 'B': (None, None)}
CONSTRAINTS = []

A_GUESSES = xrange(1, 100, 10)
B_GUESSES = xrange(10)

def get_guesses(A, B):
    guesses = [A_GUESSES, B_GUESSES]
    return list(itertools.product(*guesses))

def quick_1974((C, t), A, B):
    """
    t is duration
    C is coherence
    A is threshold coherence (given duration) for 81.6% correctness
    B is slope of psychometric curve around A

    From Kiani et al. (2008)
    """
    return 1 - 0.5*exp(-pow(C/A, B))

def log_likelihood_fcn(data, (A, B)):
    return lambda theta: -log_likelihood(data, quick_1974, (theta[0], theta[1]))

def fit(data, (A, B), quick, guesses=None, method='TNC'):
    """
    (A, B) are numerics
        the model parameters
        if None, they will be fit
    """
    thetas = (A, B)
    if guesses is None:
        guesses = get_guesses(A, B)
    bounds = [BOUNDS[key] for key, val in zip(['A', 'B'], [A, B]) if val is None]
    constraints = CONSTRAINTS
    return mle(data, log_likelihood_fcn(data, thetas), guesses, bounds, constraints, quick, method)
