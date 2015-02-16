import itertools
from collections import Iterable

import numpy as np

from mle import fit_mle, mle, APPROX_ZERO

THETA_ORDER = ['A', 'B', 'T']
A_GUESSES =  [i/10.0 for i in xrange(5, 10)]
B_GUESSES = [i/10.0 for i in xrange(5, 10)]
T_GUESSES = [1000, 500, 250, 100, 50, 10]
GUESSES = {'A': A_GUESSES, 'B': B_GUESSES, 'T': T_GUESSES}
BOUNDS = {'A': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'B': (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), 'T': (0.0+APPROX_ZERO, None)}
CONSTRAINTS = [] # [{'type': 'ineq', 'fun': lambda theta: theta[0] - theta[1]}] # A > B

DEFAULT_DELAY = 0.1/3 # 0.03

def double_saturating_exp(x, A, B, T1, T2, p, x0=DEFAULT_DELAY):
    """
    A is the end asymptote value
    B is the start asymptote value
    T1 is the 1st time constant, s.t. value is p*63.2%  of (A-B)
    T2 is the 2nd time constant, s.t. value is (1-p)*63.2%  of (A-B)
    p is the proportion of range (A-B) contributed to by T1, as opposed to T2
    """
    # print A,B,T1,T2,p,x0
    if not (0 <= p <= 1):
        return np.inf
    if T1*T2 == 0:
        return np.ones(len(x))*A if isinstance(x, Iterable) else A
    return A - p*(A-B)*np.exp(-(x-x0)*1000./T1) - (1-p)*(A-B)*np.exp(-(x-x0)*1000./T2)

def saturating_exp(x, A, B, T, x0=DEFAULT_DELAY):
    """
    A is the end asymptote value
    B is the start asymptote value
    T is the time s.t. value is 63.2%  of (A-B)
    """
    if T == 0:
        return np.ones(len(x))*A if isinstance(x, Iterable) else A
    return A - (A-B)*np.exp(-(x-x0)*1000.0/T)

def fit(data, thetas, quick=False, guesses=None, method='TNC'):
    return fit_mle(data, saturating_exp, thetas, THETA_ORDER, GUESSES, BOUNDS, CONSTRAINTS, quick, guesses, method)

def fit_df(df, B0, x0=DEFAULT_DELAY, method='L-BFGS-B'):
    """
    df is a pandas DataFrame with two columns: 'duration' (in sec) and 'correct' (bool or 0/1)
    returns (A, T) of resulting fit
    """
    F = saturating_exp
    L = lambda x, y, (A, B, T, x0), F=F: (F(x, A, B, T, x0)**y)*((1-F(x, A, B, T, x0))**(1-y))
    obj = lambda (A, T), B=B0, x=df['real_duration'], y=df['correct'], L=L: -np.log(L(x, y, (A, B, T, x0))).sum()
    guesses = [(A_GUESSES[3], T_GUESSES[3])] # list(itertools.product(A_GUESSES, T_GUESSES))
    bounds = [BOUNDS['A'], BOUNDS['T']]
    th = mle([1], obj, guesses=guesses, bounds=bounds, constraints=[], quick=False, method=method)
    if th and th[-1]['success']:
        return th[-1]['x']
    return None, None
