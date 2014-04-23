import os.path

from io import load_json
from summaries import session_grouper, coherence_grouper, as_x_y
from mle_set_B import mle_set_B

def make_curve(trials):
	groups = group_trials(trials, coherence_grouper, False)
	for coh, ts in groups.iteritems():
		rs = as_x_y(ts)

def main():
	BASEDIR = '/Users/mobeets/Dropbox/Work/Huk/temporalIntegration'
	INFILE = os.path.join(BASEDIR, 'data.json')
	TRIALS = load_json(INFILE)
	groups = group_trials(trials, session_grouper, False)

	make_curve(groups[(subj, '3d')])
	make_curve(groups[(subj, '2d')])

if __name__ == '__main__':
	main()
