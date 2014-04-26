import json
import pickle
import os.path
import logging
import argparse

import numpy as np

from dio import load_json, makefn
from session_info import DEFAULT_THETA, BINS, good_subjects, bad_sessions, good_cohs, bad_cohs
from fit_compare import pick_best_theta
from sample import sample_wr
from summaries import group_trials, subj_grouper, dot_grouper, session_grouper, coherence_grouper, as_x_y
from huk_tau_e import binned_ps, huk_tau_e
from mle_set_B import mle_set_B
from mle import mle

logging.basicConfig(level=logging.DEBUG)

def pickle_fit(results, bins, outfile, subj, cond):
    """
    n.b. json output is just for human-readability; it's not invertible since numeric keys become str
    """
    out = {}
    out['fits'] = results
    out['bins'] = bins
    out['subj'] = subj
    out['cond'] = cond
    pickle.dump(out, open(outfile, 'w'))
    json.dump(out, open(outfile.replace('.pickle', '.json'), 'w'), indent=4)

def mle_fit(ts, B, bins, coh, quick=True):
    ths = mle_set_B(ts, B=B, quick=quick)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found with fixed B={0}. Letting B vary.'.format(B)
        # logging.info(msg)
        # ths = mle(ts, quick=True)
        if not ths:
            msg = 'No fits found. Using {0}'.format(DEFAULT_THETA)
            logging.warning(msg)
            th = [DEFAULT_THETA['A'], DEFAULT_THETA['T']]
    msg = '{0}% MLE: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': th[0], 'B': B if len(th) == 2 else th[1], 'T': th[-1]}

def huk_fit(ts, B, bins, coh):
    ths, A = huk_tau_e(ts, B=B, durs=bins)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found for Huk Using tau=0.0001.'
        logging.warning(msg)
        th = [DEFAULT_THETA['T']]
    msg = '{0}% HUK: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': A, 'B': B, 'T': th[0]}

def fit_curves(trials, bins, B):
    groups = group_trials(trials, coherence_grouper, False)
    cohs = sorted(groups)
    results = {}
    for coh in cohs:
        ts = groups[coh]
        ts_cur_coh = as_x_y(ts)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        results[coh] = {}
        results[coh]['mle'] = mle_fit(ts_cur_coh, B, bins, coh)
        results[coh]['huk'] = huk_fit(ts_cur_coh, B, bins, coh)
        results[coh]['binned'] = binned_ps(ts_cur_coh, bins)
        results[coh]['ntrials'] = len(ts_cur_coh)
    return results

def fit_session_curves(trials, bins, subj, cond, pickle_outfile):
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, cond)
    logging.info(msg)
    if not trials:
        logging.info('No graphs.')
        return
    results = fit_curves(trials, bins, DEFAULT_THETA['B'])
    pickle_fit(results, bins, pickle_outfile, subj, cond)
    return results

def remove_bad_trials_by_session(trials, cond):
    """
    assumes filtering already done by cond
    """
    subjs = good_subjects[cond]
    bad_sess = bad_sessions[cond]
    keep = lambda t: t.session.subject in subjs and t.session.index not in bad_sess
    return [t for t in trials if keep(t)]

def sample_trials_by_session(trials, cond, mult=5):
    """
    assumes filtering already done by cond
    """
    groups = group_trials(trials, subj_grouper, False)
    n = min(len(ts) for ts in groups.values())
    ts_all = []
    for subj, ts in groups.iteritems():
        ts_cur = sample_wr(ts, mult*n)
        ts_all.extend(ts_cur)
    msg = 'Created {0} trials for {1} dots (sampling with replacement per subject)'.format(len(ts_all), cond)
    logging.info(msg)
    return ts_all

def remove_trials_by_coherence(trials, cond):
    bad = False
    if bad:
        cohs = bad_cohs[cond]
        return [t for t in trials if t.coherence not in cohs]
    else:
        cohs = good_cohs[cond]
        return [t for t in trials if t.coherence in cohs]

def fit(subj, cond, trials, bins, outfile, resample=False):
    trials = remove_bad_trials_by_session(trials, cond)
    trials = remove_trials_by_coherence(trials, cond)
    if resample:
        trials = sample_trials_by_session(trials, cond)
    return fit_session_curves(trials, bins, subj, cond, outfile)

def by_subject(trials, conds, bins, outdir):
    groups = group_trials(trials, session_grouper, False)
    results = {}
    for cond in conds:
        for subj in good_subjects[cond]:
            trials = groups[(subj, cond)]
            outfile = makefn(outdir, subj, cond, 'fit', 'pickle')
            fit(subj, cond, trials, bins, outfile)

def across_subjects(trials, conds, bins, outdir):
    groups = group_trials(trials, dot_grouper, False)
    subj = 'ALL'
    for cond in conds:
        trials = groups[cond]
        outfile = makefn(outdir, subj, cond, 'fit', 'pickle')
        fit(subj, cond, trials, bins, outfile, resample=True)

def main(conds, kind, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)
    TRIALS = load_json(INFILE)
    if kind == 'ALL':
        bins = list(np.logspace(np.log10(min(BINS)), np.log10(max(BINS)), 50)) # lots of bins for fun
        bins[0] = BINS[0] # to ensure lower bound on data
        across_subjects(TRIALS, conds, bins, OUTDIR)
    elif kind == 'SUBJECT':
        by_subject(TRIALS, conds, BINS, OUTDIR)
    else:
        msg = "kind {0} not recognized".format(kind)
        logging.error(msg)
        raise Exception(msg)

parser = argparse.ArgumentParser()
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
parser.add_argument('-c', "--conds", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="The number of imperatives to generate.")
parser.add_argument('-k', "--kind", default='SUBJECT', type=str, choices=['SUBJECT', 'ALL'], help="SUBJECT fits for each subject, ALL combines data and fits all at once.")
args = parser.parse_args()

if __name__ == '__main__':
    main(args.conds, args.kind, args.outdir)
