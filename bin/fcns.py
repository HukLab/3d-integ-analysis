from math import isnan
from numpy import exp

import logging
logging.basicConfig(level=logging.DEBUG)

def saturating_exp(x, A, B, T):
	"""
	A is the end asymptote value
	B is the start asymptote value
	T is the time s.t. value is 63.2%  of (A-B)
	"""
	# if T == 0 or isnan(T):
	# 	return float('inf')
	return A - (A-B)*exp(-x*1000.0/T)
	# try:
	# 	return A - (A-B)*exp(-x*1000.0/T)
	# except:
	# 	msg = (T, x, A, B)
	# 	logging.error(str(msg))
	# 	return float('inf')

def twin_limb(x, x0, S0, p):
	"""
	p is the slope at which the value increases
	x0 is the time at which the value plateaus
	S0 is the value after the value plateaus

	twin-limb function [Burr, Santoto (2001)]
	    S(x) =
	            {
	                S0 * (x/x0)^p | x < x0
	                S0             | x >= x0
	            }
	"""
	return S0 if x >= x0 else S0*pow(x/x0, p)
