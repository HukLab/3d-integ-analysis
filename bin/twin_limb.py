import logging
import itertools

from mle import mle, log_likelihood, APPROX_ZERO

logging.basicConfig(level=logging.DEBUG)

BOUNDS = {'X0': (0.0+APPROX_ZERO, None), 'S0': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'P': (None, None)}
CONSTRAINTS = []

X0_GUESSES =  xrange(1, 2000, 50)
S0_GUESSES =  [i/10.0 for i in xrange(1, 10)]
P_GUESSES =  xrange(10)

def get_guesses(X0, S0, P):
    guesses = [X0_GUESSES, S0_GUESSES, P_GUESSES]
    return list(itertools.product(*guesses))

def twin_limb(x, x0, S0, p):
    """
    p is the slope at which the value increases
    x0 is the time at which the value plateaus
    S0 is the value after the value plateaus, 0 <= S0 <= 1

    twin-limb function [Burr, Santoto (2001)]
        S(x) =
                {
                    S0 * (x/x0)^p | x < x0
                    S0            | x >= x0
                }
    """
    return S0 if x >= x0 else S0*pow(x*1.0/x0, p)

def log_likelihood_fcn(data, (X0, S0, P)):
    return lambda theta: -log_likelihood(data, twin_limb, (theta[0], theta[1], theta[2]))

def fit(data, (X0, S0, P), quick, guesses=None, method='TNC'):
    """
    (X0, S0, P) are numerics
        the model parameters
        if None, they will be fit
    """
    thetas = (X0, S0, P)
    if guesses is None:
        guesses = get_guesses(X0, S0, P)
    bounds = [BOUNDS[key] for key, val in zip(['X0', 'S0', 'P'], [X0, S0, P]) if val is None]
    constraints = CONSTRAINTS
    return mle(data, log_likelihood_fcn(data, thetas), guesses, bounds, constraints, quick, method)
