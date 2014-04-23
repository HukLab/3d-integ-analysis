import csv
import json
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

from dio import load_json
from fcns import pc_per_dur_by_coh
from fit_compare import pick_best_theta
from summaries import group_trials, session_grouper, coherence_grouper, as_x_y
from huk_tau_e import binned_ps, BINS
from mle_set_B import mle_set_B

logging.basicConfig(level=logging.DEBUG)

def make_curves(results, bins, outfile):
	min_dur, max_dur = min(bins), max(bins)
	xs = np.logspace(np.log10(min_dur), np.log10(max_dur))
	yf = lambda x: pc_per_dur_by_coh(x, th['A'], th['B'], th['T'])

	cohs = sorted(results.keys())
	nrows = 3
	ncols = int((len(cohs)-0.5)/nrows)+1
	sec_to_ms = lambda xs: [x*1000 for x in xs]

	plt.clf()
	for i, coh in enumerate(cohs):
		plt.subplot(ncols, nrows, i+1)
		plt.title('{0}% coherence'.format(int(coh*100)))
		plt.xlabel('duration (ms)')
		plt.ylabel('% correct')
		th = results[coh]['theta']
		ys = yf(xs)
		xs_binned, ys_binned = zip(*results[coh]['ps'].iteritems())
		plt.plot(sec_to_ms(xs), ys, '-')
		plt.plot(sec_to_ms(xs_binned), ys_binned, 'o')
		plt.xscale('log')
		# plt.xlim()
		plt.ylim(0.2, 1.1)
	try:
		plt.tight_layout()
	except:
		pass
	plt.savefig(outfile)

def fit_curves(trials, bins, b=0.5):
	groups = group_trials(trials, coherence_grouper, False)
	cohs = sorted(groups)
	results = {}
	for coh in cohs:
		if coh < 0.9:
			continue
		ts = groups[coh]
		ts_cur_coh = as_x_y(ts)
		ths = mle_set_B(ts_cur_coh, B=b, quick=True)
		if ths:
			th = pick_best_theta(ths)
		else:
			msg = 'No fits found. Using alpha=1.0, tau=0.0001'
			logging.warning(msg)
			th = [1.0, 0.001]
		ps = binned_ps(ts_cur_coh, bins)
		msg = '{0}%: {1}'.format(int(coh*100), th)
		logging.info(msg)
		results[coh] = {}
		results[coh]['theta'] = {'A': th[0], 'B': b, 'T': th[-1]}
		results[coh]['ps'] = ps
	return results

def write_fit_json(results, bins, outfile):
	out = {}
	out['results'] = results
	out['duration_bins'] = bins
	json.dump(out, open(outfile, 'w'))

def write_fit_csv(results, bins, outfile):
	with open(outfile, 'wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter='\t')
		th_keys = ['A', 'B', 'T']
		header = ['coh'] + th_keys + bins

		cohs = sorted(results.keys())
		header = ['cohs:'] + cohs
		csvwriter.writerow(header)

		# write thetas per coh
		for key in th_keys:
			row = [key]
			for coh in cohs:
				val = results[coh]['theta'][key]
				row.append(val)
			csvwriter.writerow(row)

		# write binned ps per coh
		for dur in bins:
			row = [dur]
			for coh in cohs:
				d = results[coh]['ps']
				if dur in d:
					val = d[dur]
				else:
					val = 'N/A'
				row.append(val)
			csvwriter.writerow(row)

def make_curve_for_session(groups, bins, subj, cond, outfile):
	trials = groups[(subj, cond)]
	msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, cond)
	logging.info(msg)
	if not trials:
		logging.info('No graphs.')
		return
	results = fit_curves(trials, bins)
	make_curves(results, bins, outfile)
	# write_fit_csv(results, bins, outfile)
	# write_fit_json(results, bins, outfile)
	# make_curve(groups[(subj, '3d')], bins, (0.0, max_dur))

def main():
	"""
	NO FITS:
		klb 2d: 100%
		huk 3d: 0% --> fixed by run with TNC solver
		lkc 3d: 100%
		krm 2d: 50%, 100%

	What about plotting, for each coh, the time it takes fcn to get to X% of (A-B)? 
	"""
	CURDIR = os.path.dirname(os.path.abspath(__file__))
	BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
	INFILE = os.path.join(BASEDIR, 'data.json')
	OUTDIR = os.path.join(BASEDIR, 'res')
	TRIALS = load_json(INFILE)
	groups = group_trials(TRIALS, session_grouper, False)

	subjs = ['klb'] # ['krm', 'klb', 'lkc', 'lnk', 'ktz', 'huk']
	conds = ['2d'] # ['2d', '3d']
	for subj in subjs:
		for cond in conds:
			if (subj, cond) not in groups:
				logging.warning('{0}, {1} not in trials!'.format(subj, cond))
				continue
			outfile = os.path.join(OUTDIR, '{0}-{1}.png'.format(subj, cond))
			make_curve_for_session(groups, BINS, subj, cond, outfile)


if __name__ == '__main__':
	main()
