import json
import pickle
import os.path
import logging
import argparse

import numpy as np

from dio import load_json, makefn
from session_info import DEFAULT_THETA, BINS, NBOOTS, NBOOTS_BINNED_PS, FIT_IS_COHLESS, all_subjs, good_subjects, bad_sessions, good_cohs, bad_cohs, QUICK_FIT, THETAS_TO_FIT, min_dur, max_dur
from mle import pick_best_theta, generic_fit
from sample import sample_wr, bootstrap
from summaries import group_trials, subj_grouper, dot_grouper, session_grouper, coherence_grouper, as_x_y, as_C_x_y
from huk_tau_e import binned_ps, huk_tau_e
import saturating_exponential
import drift_diffuse
import quick_1974
import twin_limb

logging.basicConfig(level=logging.DEBUG)

def pickle_fit(results, bins, outfile, subj, dotmode):
    """
    n.b. json output is just for human-readability; it's not invertible since numeric keys become str
    """
    out = {}
    out['fits'] = results['fits']
    out['ntrials'] = results['ntrials']
    out['cohs'] = results['cohs']
    out['bins'] = bins
    out['subj'] = subj
    out['dotmode'] = dotmode
    pickle.dump(out, open(outfile, 'w'))
    json.dump(out, open(outfile.replace('.pickle', '.json'), 'w'), indent=4)

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

def huk_fit(ts, bins, coh, guesses=None):
    fit_found = True
    B = DEFAULT_THETA['huk']['B']
    ths, A = huk_tau_e(ts, B=B, durs=bins, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        th = [DEFAULT_THETA['huk']['T']]
        msg = 'No fits found for huk-fit. Using tau={0}.'.format(th[0])
        logging.warning(msg)
        fit_found = False
    msg = '{0}% HUK: {1}'.format(int(coh*100), th)
    # logging.info(msg)
    return {'A': A, 'B': B, 'T': th[0]}, th, fit_found

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
    fit_wrapper = lambda x,y,z,w: (lambda ts, bins, coh, gs: generic_fit(x,y,z,w, QUICK_FIT, ts, bins, coh, gs))
    FIT_FCNS = {
        'drift': fit_wrapper(drift_diffuse.fit, drift_diffuse.THETA_ORDER, THETAS_TO_FIT['drift'], DEFAULT_THETA['drift']),
        # 'drift': fit_wrapper(drift_diffuse.fit_2, drift_diffuse.THETA_ORDER, THETAS_TO_FIT['drift'], DEFAULT_THETA['drift']),
        'quick_1974': fit_wrapper(quick_1974.fit, quick_1974.THETA_ORDER, THETAS_TO_FIT['quick_1974'], DEFAULT_THETA['quick_1974']),
        'sat-exp': fit_wrapper(saturating_exponential.fit, saturating_exponential.THETA_ORDER, THETAS_TO_FIT['sat-exp'], DEFAULT_THETA['sat-exp']),
        'twin-limb': fit_wrapper(twin_limb.fit, twin_limb.THETA_ORDER, THETAS_TO_FIT['twin-limb'], DEFAULT_THETA['twin-limb']),
        'huk': huk_fit,
    }

    for key in fits_to_fit:
        if fits_to_fit[key] and FIT_IS_COHLESS[key]:
            results['fits'][key] = bootstrap_fit_curves(ts_all, make_bootstrap_fcn(FIT_FCNS[key], 0))
        elif fits_to_fit[key]:
            results['fits'][key] = {}

    for coh in cohs:
        ts = groups[coh]
        ts_cur_coh = as_x_y(ts)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        for key in fits_to_fit:
            if fits_to_fit[key] and not FIT_IS_COHLESS[key]:
                results['fits'][key][coh] = bootstrap_fit_curves(ts_cur_coh, make_bootstrap_fcn(FIT_FCNS[key], coh))
        results['fits']['binned_pcor'][coh] = binned_ps(ts_cur_coh, bins, NBOOTS_BINNED_PS, include_se=True)
        results['ntrials'][coh] = len(ts_cur_coh)
    return results

def fit_session_curves(trials, bins, subj, dotmode, fits_to_fit, pickle_outfile):
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, dotmode)
    logging.info(msg)
    if not trials:
        logging.info('No graphs.')
        return
    results = fit_curves(trials, bins, fits_to_fit)
    pickle_fit(results, bins, pickle_outfile, subj, dotmode)
    return results

def remove_bad_trials_by_session(trials, dotmode):
    """
    assumes filtering already done by dotmode
    """
    subjs = good_subjects[dotmode]
    bad_sess = bad_sessions[dotmode]
    keep = lambda t: t.session.subject in subjs and t.session.index not in bad_sess
    return [t for t in trials if keep(t)]

def sample_trials_by_session(trials, dotmode, mult=5):
    """
    assumes filtering already done by dotmode
    """
    groups = group_trials(trials, subj_grouper, False)
    n = min(len(ts) for ts in groups.values())
    ts_all = []
    for subj, ts in groups.iteritems():
        ts_cur = sample_wr(ts, mult*n)
        ts_all.extend(ts_cur)
    msg = 'Created {0} trials for {1} dots (sampling with replacement per subject)'.format(len(ts_all), dotmode)
    logging.info(msg)
    return ts_all

def remove_trials_by_duration(trials, dotmode):
    return [t for t in trials if min_dur <= t.duration <= max_dur]

def remove_trials_by_coherence(trials, dotmode):
    cohs = good_cohs[dotmode]
    return [t for t in trials if t.coherence in cohs]

def fit(subj, dotmode, fits_to_fit, trials, bins, outfile, resample=False):
    trials = remove_bad_trials_by_session(trials, dotmode)
    trials = remove_trials_by_coherence(trials, dotmode)
    trials = remove_trials_by_duration(trials, dotmode)
    if resample:
        trials = sample_trials_by_session(trials, dotmode)
    return fit_session_curves(trials, bins, subj, dotmode, fits_to_fit, outfile)

def by_subject(trials, dotmodes, fits_to_fit, bins, outdir, subj=None):
    groups = group_trials(trials, session_grouper, False)
    results = {}
    for dotmode in dotmodes:
        if subj:
            subjs = [subj]
        else:
            subjs = good_subjects[dotmode]
        for cur_subj in subjs:
            trials = groups[(cur_subj, dotmode)]
            outfile = makefn(outdir, cur_subj, dotmode, 'fit', 'pickle')
            fit(cur_subj, dotmode, fits_to_fit, trials, bins, outfile)

def across_subjects(trials, dotmodes, fits_to_fit, bins, outdir):
    groups = group_trials(trials, dot_grouper, False)
    subj = 'ALL'
    for dotmode in dotmodes:
        trials = groups[dotmode]
        outfile = makefn(outdir, subj, dotmode, 'fit', 'pickle')
        fit(subj, dotmode, fits_to_fit, trials, bins, outfile, resample=True)

def main(dotmodes, subj, fits_to_fit, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)
    TRIALS = load_json(INFILE)
    if subj == 'ALL':
        bins = list(np.logspace(np.log10(min(BINS)), np.log10(max(BINS)), 50)) # lots of bins for fun
        bins[0] = BINS[0] # to ensure lower bound on data
        across_subjects(TRIALS, dotmodes, fits_to_fit, bins, OUTDIR)
    elif subj == 'SUBJECT':
        by_subject(TRIALS, dotmodes, fits_to_fit, BINS, OUTDIR)
    elif subj in all_subjs:
        by_subject(TRIALS, dotmodes, fits_to_fit, BINS, OUTDIR, subj=subj)
    else:
        msg = "subj {0} not recognized".format(subj)
        logging.error(msg)
        raise Exception(msg)

if __name__ == '__main__':
    """
    NOTE: See http://courses.washington.edu/matlab1/Lesson_5.html
    """
    ALL_FITS = FIT_IS_COHLESS.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
    parser.add_argument('-d', "--dotmode", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="2D or 3D or (default:) both.")
    parser.add_argument('-s', "--subj", default='SUBJECT', choices=['SUBJECT', 'ALL'] + all_subjs, type=str, help="SUBJECT fits for each subject, ALL combines data and fits all at once. Or specify subject like HUK")
    parser.add_argument('-f', "--fits", default=ALL_FITS, nargs='*', choices=ALL_FITS, type=str, help="The fitting methods you would like to use, from: {0}".format(ALL_FITS))
    args = parser.parse_args()

    def fits_fit(fits, all_fits):
        return dict((fit, fit in fits) for fit in all_fits)
    
    main(args.dotmode, args.subj, fits_fit(args.fits, ALL_FITS), args.outdir)
