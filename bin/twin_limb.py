from mle import fit_mle, log_likelihood, APPROX_ZERO

THETA_ORDER = ['X0', 'S0', 'P']
BOUNDS = {'X0': (0.0+APPROX_ZERO, None), 'S0': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'P': (None, None)}
X0_GUESSES =  [i/10.0 for i in xrange(1, 20, 4)]
S0_GUESSES =  [i/10.0 for i in xrange(1, 10, 3)]
P_GUESSES =  [i/10.0 for i in xrange(1, 10, 3)]
GUESSES = {'X0': X0_GUESSES, 'S0': S0_GUESSES, 'P': P_GUESSES}
CONSTRAINTS = []

def twin_limb(x, X0, S0, P):
    """
    S0 is the value after the value plateaus, 0 <= S0 <= 1
    X0 is the time at which the value plateaus
    P is the slope at which the value increases

    twin-limb function [Burr, Santoto (2001)]
        S(x) =
                {
                    S0 * (x/x0)^p | x < x0
                    S0            | x >= x0
                }
    """
    return S0 if x >= X0 else S0*pow(x*1.0/X0, P)

def fit(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, twin_limb, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)
