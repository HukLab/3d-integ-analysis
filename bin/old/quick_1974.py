from numpy import exp

from mle import fit_mle, APPROX_ZERO

THETA_ORDER = ['A', 'B']
BOUNDS = {'A': (0.0+APPROX_ZERO, 100.0-APPROX_ZERO), 'B': (None, None)}
A_GUESSES = xrange(1, 100, 10)
B_GUESSES = xrange(10)
GUESSES = {'A': A_GUESSES, 'B': B_GUESSES}
CONSTRAINTS = []

def quick_1974((C, t), A, B):
    """
    t is duration
    C is coherence
    A is threshold coherence (given duration) for 81.6% correctness
    B is slope of psychometric curve around A

    From Kiani et al. (2008)
    """
    return 1 - 0.5*exp(-pow(C/A, B))

def fit(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, quick_1974, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)
