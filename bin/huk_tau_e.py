import math

import numpy as np
from scipy.optimize import minimize

from sample import bootstrap

BINS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30, 0.45, 0.90, 1.30, 2.00]
NBOOTS = 1000

mean = lambda xs: sum(x for x in xs)*1.0/len(xs)

def bin_data(data, bins):
	"""
	if bins = [x_1, x_2, ..., x_n]
		then bins like [x_1, x_2), [x_2, x_3), ...
	"""
	if bins[0] != 0.0:
		bins = [0.0] + bins
	bins.append(float("inf"))

	data = sorted(data, key=lambda x: x[0])
	rs_per_dur = dict((dur, []) for dur in bins)

	i = 0
	l_bin = bins[i]
	r_bin = bins[i+1]
	for (dur, resp) in data:
		while dur >= r_bin:
			i += 1
			l_bin = bins[i]
			r_bin = bins[i+1]
		assert l_bin <= dur < r_bin
		rs_per_dur[l_bin].append(resp)
	return rs_per_dur

def find_p_per_dur(rs, nboots):
	rs = [int(r) for r in rs]
	assert all([r in [0,1] for r in rs])
	Rs = bootstrap(rs, nboots)
	bs = [mean(rs) for rs in Rs]
	return mean(bs)

def choose_A(durs, ps):
	" average of last three "
	durs = sorted(durs)
	return mean([ps[dur] for dur in durs[-3:]])

def p_fcn(x, A, B, tau):
	if tau == 0 or math.isnan(tau):
		return float('inf')
	try:
		return A - (A-B)*np.exp(-x*1.0/tau)
	except:
		print tau, x, A, B
		return float('inf')

def error(ps, A, B, tau):
	return sum([math.pow(y - p_fcn(x, A, B, tau), 2) for x, y in ps.iteritems()])

def error_fcn(ps, A, B):
	return lambda tau: error(ps, A, B, tau)

def get_guesses():
	return [i/10.0 for i in xrange(1, 20)]

def write_bins(d, outfile):
	import csv
	with open(outfile, 'wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter='\t')
		for dur, rs in d.iteritems():
			rows = [[dur, r] for r in rs]
			csvwriter.writerows(rows)

def huk_tau_e(data, durs=BINS, nboots=NBOOTS):
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
	resps_per_dur = bin_data(data, durs) # 0
	resps_per_dur = dict((dur, rs) for dur, rs in resps_per_dur.iteritems() if rs) # remove empties
	# write_bins(resps_per_dur, 'tmp.csv')
	ps = dict((dur, find_p_per_dur(rs, nboots)) for dur, rs in resps_per_dur.iteritems()) # 1
	A = choose_A(resps_per_dur.keys(), ps) # 2
	B = 0.5

	bnds = [(0.0001, None)]
	thetas = []
	for guess in get_guesses():
		theta = minimize(error_fcn(ps, A, B), guess, bounds=bnds)
		thetas.append(theta)
	return thetas, A
