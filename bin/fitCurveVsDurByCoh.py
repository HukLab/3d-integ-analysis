import json
import pickle
import os.path
import logging
import argparse

import numpy as np

from dio import load_json, makefn
from session_info import DEFAULT_THETA, BINS, NBOOTS, all_subjs, good_subjects, bad_sessions, good_cohs, bad_cohs, QUICK_FIT
from mle import pick_best_theta
from sample import sample_wr, bootstrap
from summaries import group_trials, subj_grouper, dot_grouper, session_grouper, coherence_grouper, as_x_y, as_C_x_y
from huk_tau_e import binned_ps, huk_tau_e
import saturating_exponential
import drift_diffuse

logging.basicConfig(level=logging.DEBUG)

def pickle_fit(results, bins, outfile, subj, cond):
    """
    n.b. json output is just for human-readability; it's not invertible since numeric keys become str
    """
    out = {}
    out['fits'] = results['fits']
    out['ntrials'] = results['ntrials']
    out['cohs'] = results['cohs']
    out['bins'] = bins
    out['subj'] = subj
    out['cond'] = cond
    pickle.dump(out, open(outfile, 'w'))
    json.dump(out, open(outfile.replace('.pickle', '.json'), 'w'), indent=4)

# def drift_fit_ts(groups):
#     ts2 = []
#     for coh in sorted(groups):
#         for x,y in as_x_y(groups[coh]):
#             ts2.append([[coh, x], y])
#     return ts2

def drift_fit(ts2):
    ths = drift_diffuse.fit(ts2, quick=QUICK_FIT)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['K']]
        msg = 'No fits found for drift diffusion. Using k={0}'.format(th[0])
        logging.warning(msg)
    msg = 'DRIFT: k={0}'.format(th[0])
    logging.info(msg)
    return {'K': th[0]}

def sat_exp_fit(ts, coh):
    B = DEFAULT_THETA['B']
    ths = saturating_exponential.fit(ts, (None, B, None), quick=QUICK_FIT)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found for sat-exp with fixed B={0}. Letting B vary.'.format(B)
        # logging.info(msg)
        # ths = saturating_exponential.fit(ts, (None, None, None), quick=QUICK_FIT)
        if not ths:
            msg = 'No fits found. Using {0}'.format(DEFAULT_THETA)
            logging.warning(msg)
            th = [DEFAULT_THETA['A'], DEFAULT_THETA['T']]
    msg = '{0}% SAT_EXP: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': th[0], 'B': B if len(th) == 2 else th[1], 'T': th[-1]}

def huk_fit(ts, bins, coh):
    B = DEFAULT_THETA['B']
    ths, A = huk_tau_e(ts, B=B, durs=bins)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['T']]
        msg = 'No fits found for huk-fit. Using tau={0}.'.format(th[0])
        logging.warning(msg)
    msg = '{0}% HUK: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': A, 'B': B, 'T': th[0]}

def bootstrap_fit_curves(ts, fcn, nboots=NBOOTS):
    """
    ts is trials
    fcn takes trials only as its input

    return mean and var of each parameter in bootstrapped fits
    """
    fits = [fcn(ts)]
    tssb = bootstrap(ts, nboots)
    for tsb in tssb:
        fit = fcn(tsb)
        fits.append(fit)
    return fits

def fit_curves(trials, bins):
    groups = group_trials(trials, coherence_grouper, False)
    ts_all = as_C_x_y(trials)
    cohs = sorted(groups)
    results = {}
    results['ntrials'] = {}
    results['cohs'] = cohs
    results['fits'] = {}
    results['fits']['huk'] = {}
    results['fits']['sat-exp'] = {}
    results['fits']['binned_pcor'] = {}
    results['fits']['drift'] = bootstrap_fit_curves(ts_all, lambda ts: drift_fit(ts))

    for coh in cohs:
        ts = groups[coh]
        ts_cur_coh = as_x_y(ts)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        results['fits']['sat-exp'][coh] = bootstrap_fit_curves(ts_cur_coh, lambda ts: sat_exp_fit(ts, coh))
        results['fits']['huk'][coh] = bootstrap_fit_curves(ts_cur_coh, lambda ts: huk_fit(ts, bins, coh))
        results['fits']['binned_pcor'][coh] = bootstrap_fit_curves(ts_cur_coh, lambda ts: binned_ps(ts, bins))
        results['ntrials'][coh] = len(ts_cur_coh)
    return results

def fit_session_curves(trials, bins, subj, cond, pickle_outfile):
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, cond)
    logging.info(msg)
    if not trials:
        logging.info('No graphs.')
        return
    results = fit_curves(trials, bins)
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

def by_subject(trials, conds, bins, outdir, subj=None):
    groups = group_trials(trials, session_grouper, False)
    results = {}
    for cond in conds:
        if subj:
            subjs = [subj]
        else:
            subjs = good_subjects[cond]
        for subj in subjs:
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

def main(conds, subj, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)
    TRIALS = load_json(INFILE)
    if subj == 'ALL':
        bins = list(np.logspace(np.log10(min(BINS)), np.log10(max(BINS)), 50)) # lots of bins for fun
        bins[0] = BINS[0] # to ensure lower bound on data
        across_subjects(TRIALS, conds, bins, OUTDIR)
    elif subj == 'SUBJECT':
        by_subject(TRIALS, conds, BINS, OUTDIR)
    elif subj in all_subjs:
        by_subject(TRIALS, conds, BINS, OUTDIR, subj=subj)
    else:
        msg = "subj {0} not recognized".format(subj)
        logging.error(msg)
        raise Exception(msg)

parser = argparse.ArgumentParser()
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
parser.add_argument('-c', "--conds", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="The number of imperatives to generate.")
parser.add_argument('-s', "--subj", default='SUBJECT', choices=['SUBJECT', 'ALL'] + all_subjs, type=str, help="SUBJECT fits for each subject, ALL combines data and fits all at once. Or specify subject like HUK")
args = parser.parse_args()

if __name__ == '__main__':
    main(args.conds, args.subj, args.outdir)
    """
    NOTE: See http://courses.washington.edu/matlab1/Lesson_5.html
    """
