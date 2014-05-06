import math
import logging

import numpy as np
from scipy.optimize import minimize

from fcns import saturating_exp, log_likelihood, keep_solution, APPROX_ZERO

logging.basicConfig(level=logging.DEBUG)

def log_likelihood_fcn(data, A, B):
	return lambda theta: -log_likelihood(data, saturating_exp, (A, B, theta[0]))

def get_guesses():
	return  [10, 50, 100, 250, 500, 1000]

def mle_set_A_B(data, A=None, B=0.5, quick=False):
	"""
	data is list [(dur, resp)]
		dur is floar
		resp is True/False
	"""
	if A is None:
		msg = 'A not supplied to mle_set_A_B'
		logging.error(msg)
		raise Exception(msg)
	# bnds = [(0.0, None)]
	bnds = [(0.0+APPROX_ZERO, None)]

	thetas = []
	ymin = float('inf')
	for guess in get_guesses():
		theta = minimize(log_likelihood_fcn(data, A, B), guess, method='TNC', bounds=bnds)
		if keep_solution(theta, bnds, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			msg = '{0}, {1}'.format(theta['x'], theta['fun'])
			logging.info(msg)
	return thetas
