import math
import logging

import numpy as np
from scipy.optimize import minimize

from session_info import BINS
from fcns import saturating_exp
from sample import bootstrap

logging.basicConfig(level=logging.DEBUG)
NBOOTS = 1000

mean = lambda xs: sum(x for x in xs)*1.0/len(xs)

def bin_data(data, bins):
	"""
	if bins = [x_1, x_2, ..., x_n]
		then bins like [x_1, x_2), [x_2, x_3), ...
	"""
	assert bins == sorted(bins)
	assert not(any([x for x,y in data if x < bins[0] or x > bins[-1]]))
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

def binned_ps(data, durs, nboots=NBOOTS):
	resps_per_dur = bin_data(data, durs) # 0
	resps_per_dur = dict((dur, rs) for dur, rs in resps_per_dur.iteritems() if rs) # remove empties
	ps = dict((dur, find_p_per_dur(rs, nboots)) for dur, rs in resps_per_dur.iteritems())
	return ps

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

def error(ps, A, B, tau):
	return sum([math.pow(y - saturating_exp(x, A, B, tau), 2) for x, y in ps.iteritems()])

def error_fcn(ps, A, B):
	return lambda tau: error(ps, A, B, tau)

def get_guesses():
	return [10, 50, 100, 250, 500, 1000]

def write_bins(d, outfile):
	import csv
	with open(outfile, 'wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter='\t')
		for dur, rs in d.iteritems():
			rows = [[dur, r] for r in rs]
			csvwriter.writerows(rows)

def huk_tau_e(data, B=0.5, durs=BINS, nboots=NBOOTS):
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
	ps = binned_ps(data, durs, nboots)
	A = choose_A(ps.keys(), ps) # 2

	bnds = [(0.0001, None)]
	thetas = []
	for guess in get_guesses():
		theta = minimize(error_fcn(ps, A, B), guess, bounds=bnds)
		thetas.append(theta)
	return thetas, A
