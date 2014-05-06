import math

import numpy as np
from scipy.optimize import minimize

from fcns import saturating_exp

APPROX_ZERO = 0.0001

def log_likelihood(arr, (A, B, T), try_harder=False):
	log_like = lambda row: saturating_exp(row[0], A, B, T)
	"""
	if try_harder:
		log_like = log_like.*0.99 + 0.005;
	"""
	arr_1 = np.vstack((arr.T, map(log_like, arr))).T
	log_likes = lambda row: np.log(row[2]) if row[1] else np.log(1-row[2])
	val = sum(map(log_likes, arr_1))
	"""
	# should check exactly which of these cases can happen:
	if val == float('inf') or val == float('-inf') or math.isnan(val):
		if not try_harder:
			val = log_likelihood(arr, (A, B, T), try_harder=True)
	return val
	"""
	return val

def log_likelihood_fcn(data):
	return lambda theta: -log_likelihood(data, (theta[0], theta[1], theta[2]))

def get_guesses():
	ags =  [i/10.0 for i in xrange(1, 10)]
	bgs = [10, 50, 100, 250, 500, 1000]
	tgs = [i/10.0 for i in xrange(1, 10)] # no clue yet

	# ags = [0.9]
	# bgs = [0.6]
	# tgs = [0.9]
	guesses = [(a, b, t) for a in ags for b in bgs for t in tgs]
	return guesses

def decide_to_keep(theta, ymin):
	if theta['success']:
		close_enough = lambda x,y: abs(x-y) < 0.001
		if close_enough(theta['x'][0], APPROX_ZERO) or close_enough(theta['x'][0], 1.0-APPROX_ZERO):
			return False
		elif close_enough(theta['x'][1], APPROX_ZERO) or close_enough(theta['x'][1], 1.0-APPROX_ZERO):
			return False
		elif close_enough(theta['x'][2], APPROX_ZERO):
			return False
		if theta['fun'] < ymin:
			return True
	return False

def mle(data, quick=False):
	"""
	data is list [(dur, resp)]
		dur is floar
		resp is True/False
	"""
	# data = np.array([(x, int(y)) for x,y in data])
	bnds = [(APPROX_ZERO, 1.0-APPROX_ZERO), (APPROX_ZERO, 1.0-APPROX_ZERO), (APPROX_ZERO, None)]
	cons = [{'type': 'ineq', 'fun': lambda theta: theta[0] - theta[1] - APPROX_ZERO}] # A > B

	thetas = []
	ymin = float('inf')
	for guess in get_guesses():
		# bounds only for: L-BFGS-B, TNC, SLSQP
		theta = minimize(log_likelihood_fcn(data), guess, method='TNC', bounds=bnds, constraints=cons)
		if decide_to_keep(theta, ymin):
			ymin = theta['fun']
			thetas.append(theta)
			if quick:
				return thetas
			msg = '{0}, {1}'.format(theta['x'], theta['fun'])
			logging.info(msg)
	return thetas
