import json
import pickle
import os.path
import logging
import argparse

import numpy as np

from dio import load_json, makefn
from session_info import DEFAULT_THETA, BINS, NBOOTS, NBOOTS_BINNED_PS, FIT_IS_PER_COH, all_subjs, good_subjects, bad_sessions, good_cohs, bad_cohs, QUICK_FIT
from mle import pick_best_theta
from sample import sample_wr, bootstrap
from summaries import group_trials, subj_grouper, dot_grouper, session_grouper, coherence_grouper, as_x_y, as_C_x_y
from huk_tau_e import binned_ps, huk_tau_e
import saturating_exponential
import drift_diffuse
import quick_1974
import twin_limb

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

def quick_1974_fit(ts2, guesses=None):
    fit_found = True
    ths = quick_1974.fit(ts2, (None, None), quick=QUICK_FIT, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['quick_1974']['A'], DEFAULT_THETA['quick_1974']['B']]
        msg = 'No fits found for drift diffusion. Using A={0}, B={1}'.format(th[0], th[1])
        fit_found = False
        # logging.warning(msg)
    msg = 'QUICK_1974: A={0}, B={1}'.format(th[0], th[1])
    # logging.info(msg)
    return {'A': th[0], 'B': th[1]}, th, fit_found

def drift_fit(ts2, guesses=None):
    fit_found = True
    X0 = DEFAULT_THETA['drift']['X0']
    ths = drift_diffuse.fit(ts2, (None, X0), quick=QUICK_FIT, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['drift']['K']]
        msg = 'No fits found for drift diffusion. Using K={0}'.format(th[0])
        fit_found = False
        # logging.warning(msg)
    msg = 'DRIFT: K={0}'.format(th[0])
    # logging.info(msg)
    return {'K': th[0], 'X0': X0}, th, fit_found

def twin_limb_fit(ts, bins, coh, guesses=None):
    fit_found = True
    ths = twin_limb.fit(ts, (None, None, None), quick=QUICK_FIT, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['twin-limb']['X0'], DEFAULT_THETA['twin-limb']['S0'], DEFAULT_THETA['twin-limb']['P']]
        msg = 'No fits found for drift diffusion. Using X0={0}, S0={1}, P={2}'.format(th[0], th[1], th[2])
        fit_found = False
        # logging.warning(msg)
    msg = 'DRIFT: X0={0}, S0={1}, P={2}'.format(th[0], th[1], th[2])
    # logging.info(msg)
    return {'X0': th[0], 'S0': th[1], 'P': th[2]}, th, fit_found

def sat_exp_fit(ts, bins, coh, guesses=None):
    fit_found = True
    B = DEFAULT_THETA['huk']['B']
    ths = saturating_exponential.fit(ts, (None, B, None), quick=QUICK_FIT, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found for sat-exp with fixed B={0}. Letting B vary.'.format(B)
        # logging.info(msg)
        # ths = saturating_exponential.fit(ts, (None, None, None), quick=QUICK_FIT)
        if not ths:
            msg = 'No fits found. Using {0}'.format(DEFAULT_THETA['sat-exp'])
            # logging.warning(msg)
            fit_found = False
            th = [DEFAULT_THETA['sat-exp']['A'], DEFAULT_THETA['sat-exp']['T']]
    msg = '{0}% SAT_EXP: {1}'.format(int(coh*100), th)
    # logging.info(msg)
    return {'A': th[0], 'B': B if len(th) == 2 else th[1], 'T': th[-1]}, th, fit_found

def huk_fit(ts, bins, coh, guesses=None):
    fit_found = True
    B = DEFAULT_THETA['huk']['B']
    ths, A = huk_tau_e(ts, B=B, durs=bins, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['huk']['T']]
        msg = 'No fits found for huk-fit. Using tau={0}.'.format(th[0])
        # logging.warning(msg)
        fit_found = False
    msg = '{0}% HUK: {1}'.format(int(coh*100), th)
    # logging.info(msg)
    return {'A': A, 'B': B, 'T': th[0]}, th, fit_found

def bootstrap_fit_curves(ts, fit_fcn, nboots=NBOOTS):
    """
    ts is trials
    fcn takes trials and guesses as its inputs

    return mean and var of each parameter in bootstrapped fits
    """
    use_as_guess = lambda fit: [[float("%.3f" % x) for x in fit[1]]]
    og_fit = fit_fcn(ts, None)
    guess = use_as_guess(og_fit) if og_fit[2] else None # will use initial solution as guess
    bootstrapped_fits = []
    if nboots > 0:
        logging.info('Bootstrapping {0} time(s)'.format(nboots))
    for tsb in bootstrap(ts, nboots):
        f = fit_fcn(tsb, guess)
        bootstrapped_fits.append(f)
        if guess is None and f[2]:
            guess = use_as_guess(f)
    fits = [og_fit] + bootstrapped_fits
    fits = [fit for fit, raw, success in fits if success] # remove unsuccessful attempts
    if not fits:
        fits = [og_fit[0]]
    return fits

def fit_curves(trials, bins, fits_to_fit):
    groups = group_trials(trials, coherence_grouper, False)
    ts_all = as_C_x_y(trials)
    cohs = sorted(groups)
    results = {}
    results['ntrials'] = {}
    results['cohs'] = cohs
    results['fits'] = {}
    results['fits']['binned_pcor'] = {}

    make_bootstrap_fcn = lambda fcn, coh: lambda ts, gs: fcn(ts, bins, coh, gs)
    make_bootstrap_fcn_nocoh = lambda fcn: lambda ts, gs: fcn(ts, gs)
    FIT_FCNS = {'drift': drift_fit, 'quick_1974': quick_1974_fit, 'huk': huk_fit, 'sat-exp': sat_exp_fit, 'twin-limb': twin_limb_fit}

    for key in fits_to_fit:
        if fits_to_fit[key] and FIT_IS_PER_COH[key]:
            results['fits'][key] = bootstrap_fit_curves(ts_all, make_bootstrap_fcn_nocoh(FIT_FCNS[key]))
        elif fits_to_fit[key]:
            results['fits'][key] = {}

    for coh in cohs:
        ts = groups[coh]
        ts_cur_coh = as_x_y(ts)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        for key in fits_to_fit:
            if fits_to_fit[key] and not FIT_IS_PER_COH[key]:
                results['fits'][key][coh] = bootstrap_fit_curves(ts_cur_coh, make_bootstrap_fcn(FIT_FCNS[key], coh))
        results['fits']['binned_pcor'][coh] = binned_ps(ts_cur_coh, bins, NBOOTS_BINNED_PS, include_se=True)
        results['ntrials'][coh] = len(ts_cur_coh)
    return results

def fit_session_curves(trials, bins, subj, cond, fits_to_fit, pickle_outfile):
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, cond)
    logging.info(msg)
    if not trials:
        logging.info('No graphs.')
        return
    results = fit_curves(trials, bins, fits_to_fit)
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

def fit(subj, cond, fits_to_fit, trials, bins, outfile, resample=False):
    trials = remove_bad_trials_by_session(trials, cond)
    trials = remove_trials_by_coherence(trials, cond)
    if resample:
        trials = sample_trials_by_session(trials, cond)
    return fit_session_curves(trials, bins, subj, cond, fits_to_fit, outfile)

def by_subject(trials, conds, fits_to_fit, bins, outdir, subj=None):
    groups = group_trials(trials, session_grouper, False)
    results = {}
    for cond in conds:
        if subj:
            subjs = [subj]
        else:
            subjs = good_subjects[cond]
        for cur_subj in subjs:
            trials = groups[(cur_subj, cond)]
            outfile = makefn(outdir, cur_subj, cond, 'fit', 'pickle')
            fit(cur_subj, cond, fits_to_fit, trials, bins, outfile)

def across_subjects(trials, conds, fits_to_fit, bins, outdir):
    groups = group_trials(trials, dot_grouper, False)
    subj = 'ALL'
    for cond in conds:
        trials = groups[cond]
        outfile = makefn(outdir, subj, cond, 'fit', 'pickle')
        fit(subj, cond, fits_to_fit, trials, bins, outfile, resample=True)

def main(conds, subj, fits_to_fit, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)
    TRIALS = load_json(INFILE)
    if subj == 'ALL':
        bins = list(np.logspace(np.log10(min(BINS)), np.log10(max(BINS)), 50)) # lots of bins for fun
        bins[0] = BINS[0] # to ensure lower bound on data
        across_subjects(TRIALS, conds, fits_to_fit, bins, OUTDIR)
    elif subj == 'SUBJECT':
        by_subject(TRIALS, conds, fits_to_fit, BINS, OUTDIR)
    elif subj in all_subjs:
        by_subject(TRIALS, conds, fits_to_fit, BINS, OUTDIR, subj=subj)
    else:
        msg = "subj {0} not recognized".format(subj)
        logging.error(msg)
        raise Exception(msg)

ALL_FITS = FIT_IS_PER_COH.keys()
parser = argparse.ArgumentParser()
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
parser.add_argument('-c', "--conds", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="2D or 3D or both.")
parser.add_argument('-s', "--subj", default='SUBJECT', choices=['SUBJECT', 'ALL'] + all_subjs, type=str, help="SUBJECT fits for each subject, ALL combines data and fits all at once. Or specify subject like HUK")
parser.add_argument('-f', "--fits", default=ALL_FITS, nargs='*', choices=ALL_FITS, type=str, help="The fitting methods you would like to use, from: {0}".format(ALL_FITS))
args = parser.parse_args()

def fits_fit(fits, all_fits):
    return dict((fit, fit in fits) for fit in all_fits)

if __name__ == '__main__':
    main(args.conds, args.subj, fits_fit(args.fits, ALL_FITS), args.outdir)
    """
    NOTE: See http://courses.washington.edu/matlab1/Lesson_5.html
    """
