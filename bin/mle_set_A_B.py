import math
import logging

import numpy as np
from scipy.optimize import minimize

from mle import APPROX_ZERO, log_likelihood

logging.basicConfig(level=logging.DEBUG)

def log_likelihood_fcn(data, A, B):
	return lambda theta: -log_likelihood(data, (A, B, theta[0]))

def decide_to_keep(theta, ymin):
	if theta['success']:
		close_enough = lambda x,y: abs(x-y) < 0.001
		if close_enough(theta['x'], APPROX_ZERO):
			return False
		if theta['fun'] < ymin:
			return True
	return False

def get_guesses():
	return  [10, 50, 100, 250, 500, 1000]

def mle_set_A_B(data, A=None, B=0.5, quick=False):
	"""
	data is list [(dur, resp)]
		dur is floar
		resp is True/False
	"""
	# data = np.array([(x,int(y)) for x,y in data])
	if A is None:
		msg = 'A not supplied to mle_set_A_B'
		logging.error(msg)
		raise Exception(msg)
	bnds = [(APPROX_ZERO, None)]

	thetas = []
	ymin = float('inf')
	for guess in get_guesses():
		# bounds only for: L-BFGS-B, TNC, SLSQP
		theta = minimize(log_likelihood_fcn(data, A, B), guess, method='TNC', bounds=bnds)
		if decide_to_keep(theta, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			msg = '{0}, {1}'.format(theta['x'], theta['fun'])
			logging.info(msg)
	return thetas
