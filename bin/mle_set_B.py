import math

import numpy as np
from scipy.optimize import minimize

from mle import APPROX_ZERO, log_likelihood

def log_likelihood_fcn(data, B):
	return lambda theta: -log_likelihood(data, (theta[0], B, theta[1]))

def decide_to_keep(theta, ymin):
	if theta['success']:
		close_enough = lambda x,y: abs(x-y) < 0.001
		if close_enough(theta['x'][0], APPROX_ZERO) or close_enough(theta['x'][0], 1.0-APPROX_ZERO):
			return False
		elif close_enough(theta['x'][1], APPROX_ZERO):
			return False
		if theta['fun'] < ymin:
			return True
	return False

def get_guesses():
	ags =  [i/10.0 for i in xrange(1, 10)]
	tgs = [i/10.0 for i in xrange(1, 10)] # no clue yet
	guesses = [(a, t) for a in ags for t in tgs]
	return guesses

def mle_set_B(data, B=0.5, quick=False):
	"""
	data is list [(dur, resp)]
		dur is floar
		resp is True/False
	"""
	data = np.array([(x,int(y)) for x,y in data])
	bnds = [(APPROX_ZERO, 1.0-APPROX_ZERO), (APPROX_ZERO, None)]
	cons = [{'type': 'ineq', 'fun': lambda theta: theta[0] - B}] # A > B

	thetas = []
	ymin = float('inf')
	for guess in get_guesses():
		# bounds only for: L-BFGS-B, TNC, SLSQP
		theta = minimize(log_likelihood_fcn(data, B), guess, method='L-BFGS-B', bounds=bnds, constraints=cons)
		if decide_to_keep(theta, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			print theta['x'], theta['fun']
	return thetas
