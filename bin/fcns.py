from math import isnan
from numpy import exp

import logging
logging.basicConfig(level=logging.DEBUG)

def pc_per_dur_by_coh(x, A, B, T):
	if T == 0 or isnan(T):
		return float('inf')
	try:
		return A - (A-B)*exp(-x*1.0/T)
	except:
		msg = (T, x, A, B)
		logging.error(str(msg))
		return float('inf')
