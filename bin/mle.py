import math
import itertools
import logging
from collections import Iterable

import numpy as np
from scipy.optimize import minimize

from fcns import saturating_exp, log_likelihood, keep_solution, APPROX_ZERO

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
		msg = "MLE incorrectly defined."
		logging.error(msg)
		raise Exception(msg)
	return list(itertools.product(*guesses)) # cartesian product

def log_likelihood_fcn(data, (A, B, T)):
	if A is None and B is None and T is None:
		return lambda theta: -log_likelihood(data, saturating_exp, (theta[0], theta[1], theta[2]))
	elif A is None and B is not None and T is None:
		return lambda theta: -log_likelihood(data, saturating_exp, (theta[0], B, theta[1]))
	elif A is not None and B is not None and T is None:
		return lambda theta: -log_likelihood(data, saturating_exp, (A, B, theta[0]))
	else:
		msg = "MLE incorrectly defined."
		logging.error(msg)
		raise Exception(msg)

def mle(data, (A, B, T), quick=False, method='TNC'):
	"""
	data is list [(dur, resp)]
		dur is float
		resp is bool
	(A, B, T) are numerics
		the model parameters
		if None, they will be fit
	quick is bool
		chooses the first solution not touching the bounds
	method is str
		bounds only for: L-BFGS-B, TNC, SLSQP
		constraints only for: COBYLA, SLSQP
		NOTE: SLSQP tends to give a lot of run-time errors...
	"""
	bnds = [BOUNDS[key] for key, val in zip(['A', 'B', 'T'], [A, B, T]) if val is None]
	cons = CONSTRAINTS

	thetas = []
	ymin = float('inf')
	for guess in get_guesses(A, B, T):
		theta = minimize(log_likelihood_fcn(data, (A, B, T)), guess, method=method, bounds=bnds, constraints=cons)
		if keep_solution(theta, bnds, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			msg = '{0}, {1}'.format(theta['x'], theta['fun'])
			logging.info(msg)
	return thetas
