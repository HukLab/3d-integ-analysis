from math import sqrt, erfc

from mle import APPROX_ZERO, fit_mle

THETA_ORDER = ['K', 'X0']
BOUNDS = {'K': (None, None), 'X0': (None, 0.04-APPROX_ZERO)}
GUESSES = {'K': [i/10.0 for i in xrange(1, 10)], 'X0': [0.0, 0.02, 0.04-APPROX_ZERO]}

THETA_ORDER_1 = ['K', 'L', 'X0']
BOUNDS_1 = {'K': (None, None), 'L': (APPROX_ZERO, None), 'X0': (None, 0.04-APPROX_ZERO)}
GUESSES_1 = {'K': [i/10.0 for i in xrange(1, 10)], 'L': xrange(1, 10), 'X0': [0.0, 0.02, 0.04-APPROX_ZERO]}

CONSTRAINTS = []

def drift_diffusion((C, t), K, X0):
    """
    t is duration
    C is coherence
    K relates the drift rate to C

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
    return 0.5*erfc(-K*C*(t-X0) / sqrt(2*(t-X0)))


def drift_diffusion_1((C, t), K, L, X0):
    """
    t is duration
    C is coherence
    K relates the drift rate to C
    L relates the diffusion rate to C
    """
    return 0.5*erfc(-K*C*(t-X0) / sqrt(2*(L/C)*(t-X0)))

def drift_diffusion_1b((C, t), K, L, X0):
    """
    t is duration
    C is coherence
    K relates the drift rate to C
    L is the diffusion rate (not dependent on C)
    """
    return 0.5*erfc(-K*C*(t-X0) / sqrt(2*L*(t-X0)))

def drift_diffusion_2(x, K, X0):
    """
    This is a version letting K be fit for each coherence.
    """
    return 0.5*erfc(-K*(x-X0) / sqrt(2*(x-X0)))

def drift_diffusion_2b(x, K, L, X0):
    """
    This is a version letting K and L be fit for each coherence.
    """
    return 0.5*erfc(-K*(x-X0) / sqrt(2*L*(x-X0)))

def fit(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, drift_diffusion, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)

def fit_1(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, drift_diffusion_1, thetas, THETA_ORDER_1, GUESSES_1, BOUNDS_1, CONSTRAINTS, quick, guesses, method)

def fit_1b(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, drift_diffusion_1b, thetas, THETA_ORDER_1, GUESSES_1, BOUNDS_1, CONSTRAINTS, quick, guesses, method)

def fit_2(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, drift_diffusion_2, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)

def fit_2b(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, drift_diffusion_2b, thetas, THETA_ORDER_1, GUESSES_1, BOUNDS_1, CONSTRAINTS, quick, guesses, method)
