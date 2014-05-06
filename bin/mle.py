import math
import logging

import numpy as np
from scipy.optimize import minimize

from fcns import saturating_exp, log_likelihood, keep_solution, APPROX_ZERO

logging.basicConfig(level=logging.DEBUG)

def log_likelihood_fcn(data):
	return lambda theta: -log_likelihood(data, saturating_exp, (theta[0], theta[1], theta[2]))

def get_guesses():
	ags =  [i/10.0 for i in xrange(1, 10)]
	bgs = [10, 50, 100, 250, 500, 1000]
	tgs = [i/10.0 for i in xrange(1, 10)]
	guesses = [(a, b, t) for a in ags for b in bgs for t in tgs]
	return guesses

def mle(data, quick=False):
	"""
	data is list [(dur, resp)]
		dur is floar
		resp is True/False
	"""
	# bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, None)]
	bnds = [(0.0+APPROX_ZERO, 1.0-APPROX_ZERO), (0.0+APPROX_ZERO, 1.0-APPROX_ZERO), (0.0+APPROX_ZERO, None)]
	cons = [{'type': 'ineq', 'fun': lambda theta: theta[0] - theta[1]}] # A > B

	thetas = []
	ymin = float('inf')
	for guess in get_guesses():
		# bounds only for: L-BFGS-B, TNC, SLSQP
		# constraints only for: COBYLA, SLSQP
		# SLSQP tends to give a lot of run-time errors...
		theta = minimize(log_likelihood_fcn(data), guess, method='TNC', bounds=bnds, constraints=cons)
		if keep_solution(theta, bnds, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			msg = '{0}, {1}'.format(theta['x'], theta['fun'])
			logging.info(msg)
	return thetas
