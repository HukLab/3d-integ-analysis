import os.path
import logging

from dio import load_json
from summaries import by_coherence, as_x_y
from mle import mle
# from mle_set_B import mle_set_B
# from mle_set_A_B import mle_set_A_B
from huk_tau_e import huk_tau_e

logging.basicConfig(level=logging.DEBUG)

def write_ts(data, outfile):
	import csv
	with open(outfile, 'wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter='\t')
		csvwriter.writerows(data)

def main(subj='huk', cond='2d', coh=0.12):
	BASEDIR = '/Users/mobeets/Dropbox/Work/Huk/temporalIntegration'
	INFILE = os.path.join(BASEDIR, 'data.json')
	OUTFILE = os.path.join(BASEDIR, 'huk-2d-coh12.csv')
	TRIALS = load_json(INFILE)
	trials = by_coherence(TRIALS, (subj, cond), coh)
	ts = as_x_y(trials)
	# write_ts(ts, OUTFILE)
	msg = '{0} trials total'.format(len(ts))
	logging.info(msg)
	
	# print 'HUK'
	# thetas_huk, A = huk_tau_e(ts) # might want to use actual bins for this session rather than the default
	# th_huk = pick_best_theta(thetas_huk, True)
	# print th_huk
	# print '----------'

	print 'MLE'
	thetas_mle = mle_set_A_B(ts)
	th_mle = pick_best_theta(thetas_mle, True)
	print th_mle
	print '----------'

	# import csv
	# with open('tmp.csv', 'wb') as csvfile:
	# 	csvwriter = csv.writer(csvfile, delimiter='\t')
	# 	rows_1 = ['mle'] + thetas_mle
	# 	rows_2 = ['huk'] + thetas_huk
	# 	csvwriter.writerows([rows_1, rows_2])

if __name__ == '__main__':
	main()
