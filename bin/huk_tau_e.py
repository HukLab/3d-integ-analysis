import logging

import numpy as np
from scipy.optimize import minimize

from session_info import BINS, NBOOTS_BINNED_PS
from saturating_exponential import saturating_exp
from sample import bootstrap, bootstrap_se

logging.basicConfig(level=logging.DEBUG)

def bin_data(data, bins):
	"""
	if bins = [x_1, x_2, ..., x_n]
		then bins like [x_1, x_2), [x_2, x_3), ..., [x_n-1, x_n]
	"""
	assert bins == sorted(bins)
	assert not(any([x for x,y in data if x < bins[0] or x > bins[-1]]))
	data = sorted(data, key=lambda x: x[0])
	rs_per_dur = dict((dur, []) for dur in bins)

	i = 0
	l_bin = bins[i]
	r_bin = bins[i+1]
	for (dur, resp) in data:
		while dur >= r_bin and dur != bins[-1]:
			i += 1
			l_bin = bins[i]
			r_bin = bins[i+1]
		assert l_bin <= dur < r_bin or (l_bin <= dur <= r_bin and dur == bins[-1])
		rs_per_dur[l_bin].append(resp)
	return rs_per_dur

def binned_ps(data, durs, nboots=NBOOTS_BINNED_PS, include_se=True):
	resps_per_dur = bin_data(data, durs) # 0
	resps_per_dur = dict((dur, rs) for dur, rs in resps_per_dur.iteritems() if rs) # remove empties
	return dict((dur, find_p_per_dur(rs, nboots, include_se)) for dur, rs in resps_per_dur.iteritems())

def find_p_per_dur(rs, nboots, include_se):
	assert set(rs) <= set([0.0, 1.0, 0, 1])
	Rs = bootstrap(rs, nboots)
	bs = [np.mean(rs) for rs in Rs]
	if include_se:
		return np.mean(bs), bootstrap_se(bs)
	else:
		return np.mean(bs)

def choose_A(durs, ps):
	" average of last three "
	durs = sorted(durs)
	return np.mean([ps[dur] for dur in durs[-3:]])

def error(ps, A, B, tau):
	return sum([pow(y - saturating_exp(x, A, B, tau), 2) for x, y in ps.iteritems()])

def error_fcn(ps, A, B):
	return lambda tau: error(ps, A, B, tau)

def get_guesses():
	ts = [10, 50, 100, 250, 500, 1000]
	return ts
	# return [tuple(x) for x in ts]

def write_bins(d, outfile):
	import csv
	with open(outfile, 'wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter='\t')
		for dur, rs in d.iteritems():
			rows = [[dur, r] for r in rs]
			csvwriter.writerows(rows)

def huk_tau_e(data, B=0.5, durs=BINS, guesses=None, nboots=NBOOTS_BINNED_PS):
	"""
	data is list [(dur, resp), ...]
		dur is floar
		resp is True/False
	durs is [float, ...]
	
	algorithm:
		0. bin data by dur
		1. for each binned-dur
			bootstrap and find mean of each bootstrap sample
			take mean of all bootstrap means
		2.
			A = avg of last three durations
			B = 0.5 (chance)
		3. minimize squared error
	"""
	if guesses is None:
		guesses = get_guesses()
	ps = binned_ps(data, durs, nboots, False)
	A = choose_A(ps.keys(), ps) # 2

	bnds = [(0.0001, None)]
	thetas = []
	for guess in guesses:
		theta = minimize(error_fcn(ps, A, B), guess, bounds=bnds)
		thetas.append(theta)
	return thetas, A
