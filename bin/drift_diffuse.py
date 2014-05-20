import logging
from math import sqrt, erf

from mle import mle, log_likelihood

logging.basicConfig(level=logging.DEBUG)

BOUNDS = [(None, None)]
GUESSES = [i/10.0 for i in xrange(1, 10)]
CONSTRAINTS = []

def drift_diffusion((C, t), k, X0=0.038):
    """
    t is duration
    C is coherence
    k relates the drift rate to C

    From Selen et al. (2012):
        dDV = kCdt + dW
            DV = decision variable
            C = coherence
            t = duration
            W = standard Brownian motion with unit variance over one second
        This yields (according to the paper):
            Let X := correctness variable, in {0, 1}
            f(DV) = N(kCt, t) = gaussian with mean kCt, variance t
        =>
            p(C, t) = P(X == 1 | C, t) = P(DV > 0) = 1 - F(0)
            n.b. F(0) = 0.5 + 0.5*erf(-kCt / sqrt(2*t))]
        =>
            p(C, t) = 1 - F(0) = 0.5 - 0.5*erf(-kCt / sqrt(2*t))
    """
    return 0.5 - 0.5*erf(-k*C*(t-X0) / sqrt(2*(t-X0)))

def log_likelihood_fcn(data):
    return lambda theta: -log_likelihood(data, drift_diffusion, [theta])

def fit(data, quick=False, guesses=None, method='TNC'):
    if guesses is None:
        guesses = GUESSES
    bounds = BOUNDS
    constraints = CONSTRAINTS
    return mle(data, log_likelihood_fcn(data), guesses, bounds, constraints, quick, method)
