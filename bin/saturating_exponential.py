from collections import Iterable

import numpy as np

from mle import fit_mle, APPROX_ZERO

THETA_ORDER = ['A', 'B', 'T']
A_GUESSES =  [i/10.0 for i in xrange(5, 10)]
B_GUESSES = [i/10.0 for i in xrange(5, 10)]
T_GUESSES = [1000, 500, 250, 100, 50, 10]
GUESSES = {'A': A_GUESSES, 'B': B_GUESSES, 'T': T_GUESSES}
BOUNDS = {'A': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'B': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'T': (0.0+APPROX_ZERO, None)}
CONSTRAINTS = [] # [{'type': 'ineq', 'fun': lambda theta: theta[0] - theta[1]}] # A > B

def saturating_exp(x, A, B, T):
    """
    A is the end asymptote value
    B is the start asymptote value
    T is the time s.t. value is 63.2%  of (A-B)
    """
    if T == 0:
        # returns either A or list of As depending on how whether x is numeric or array
        return np.ones(len(x))*A if isinstance(x, Iterable) else A
    return A - (A-B)*np.exp(-(x-0.039)*1000.0/T)
    # return A - (A-B)*np.exp(-x*1000.0/T)

def fit(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, saturating_exp, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)
