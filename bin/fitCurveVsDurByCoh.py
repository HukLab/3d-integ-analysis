import pickle
import os.path
import logging
import argparse

import numpy as np
import pandas as pd

from pd_io import load, resample_by_grp
from sample import sample_wr, bootstrap, bootstrap_se
from settings import DEFAULT_THETA, NBOOTS_BINNED_PS, FIT_IS_COHLESS, QUICK_FIT, THETAS_TO_FIT, AGG_SUBJ_NAME, BINS, min_dur, max_dur, NBINS_COMBINED, NBINS

# import twin_limb
# import quick_1974
import drift_diffuse
import saturating_exponential
from huk_tau_e import binned_ps, huk_tau_e
from mle import pick_best_theta, generic_fit

logging.basicConfig(level=logging.DEBUG)
makefn = lambda outdir, subj, cond, name, ext: os.path.join(outdir, 'fitCurveVsDurByCoh-{0}-{1}{2}.{3}'.format(subj, cond, name, ext))

def write_csv(res, outdir):
    a, b = [], []
    subjs = set()
    for (subj, dotmode), obj in res.iteritems():
        subjs.add(subj)
        for coh, vals in obj['fits']['binned_pcor'].iteritems():
            coh = float(coh)
            durs = sorted(vals.keys())
            for di, dur in enumerate(durs):
                xs = vals[dur]
                pc, se, n = xs
                r1 = {'subj': subj, 'dotmode': dotmode, 'coh': coh, 'di': di, 'dur': dur, 'pc': pc, 'se': se, 'ntrials': n}
                a.append(r1)
        for coh, items in obj['fits']['sat-exp'].iteritems():
            coh = float(coh)
            for bi, th in enumerate(items):
                r2 = {'subj': subj, 'dotmode': dotmode, 'bi': bi, 'coh': coh, 'A': th['A'], 'B': th['B'], 'T': th['T']}
                b.append(r2)
    subjs = list(subjs)
    subj = subjs[0] if len(subjs) == 1 else 'AGG'
    df1 = pd.DataFrame(a)
    df2 = pd.DataFrame(b)
    outfile1 = makefn(outdir, subj, '', 'pts', 'csv')
    outfile2 = makefn(outdir, subj, '', 'params', 'csv')
    df1.to_csv(outfile1)
    df2.to_csv(outfile2)
    return df1, df2

def write_pickle(results, bins, outfile, subj, dotmode):
    """
    SB1 = ['subj', 'dotmode', 'coh', 'di', 'dur', 'pc', 'se', 'ntrials']
    SB2 = ['subj', 'dotmode', 'bi', 'coh', 'A', 'B', 'T']
    """
    out = {}
    out['fits'] = results['fits']
    out['ntrials'] = results['ntrials']
    out['cohs'] = results['cohs']
    out['bins'] = bins
    out['subj'] = subj
    out['dotmode'] = dotmode
    pickle.dump(out, open(outfile, 'w'))
    return out

def bootstrap_fit_curves(ts, fit_fcn, nboots):
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

fit_wrapper = lambda x, y, z, w: (lambda ts, bins, coh, gs: generic_fit(x, y, z, w, QUICK_FIT, ts, bins, coh, gs))
FIT_FCNS = {
    'drift': fit_wrapper(drift_diffuse.fit, drift_diffuse.THETA_ORDER, THETAS_TO_FIT['drift'], DEFAULT_THETA['drift']),
    'drift-diff': fit_wrapper(drift_diffuse.fit_1, drift_diffuse.THETA_ORDER_1, THETAS_TO_FIT['drift-diff'], DEFAULT_THETA['drift-diff']),
    # 'quick_1974': fit_wrapper(quick_1974.fit, quick_1974.THETA_ORDER, THETAS_TO_FIT['quick_1974'], DEFAULT_THETA['quick_1974']),
    'sat-exp': fit_wrapper(saturating_exponential.fit, saturating_exponential.THETA_ORDER, THETAS_TO_FIT['sat-exp'], DEFAULT_THETA['sat-exp']),
    # 'twin-limb': fit_wrapper(twin_limb.fit, twin_limb.THETA_ORDER, THETAS_TO_FIT['twin-limb'], DEFAULT_THETA['twin-limb']),
    'huk': huk_fit,
}

def as_C_x_y(df):
    vals = df[['coherence', 'real_duration', 'correct']].values
    return np.array([([a, b], int(c)) for a,b,c in vals])

def as_x_y(df):
    vals = df[['real_duration', 'correct']].values
    return np.array([(a, int(b)) for a,b in vals])

durmap_fcn = lambda df: dict(df.groupby('duration_index')['real_duration'].agg(min).reset_index().values)

def fit(df, bins, fits_to_fit, nboots):
    cohs = sorted(df['coherence'].unique())
    results = {}
    results['ntrials'] = {}
    results['cohs'] = cohs
    results['fits'] = dict((fit, {}) for fit in fits_to_fit)
    results['fits']['binned_pcor'] = {}
    make_bootstrap_fcn = lambda fcn, coh: lambda ts, gs: fcn(ts, bins, coh, gs)    

    ts_all =  as_C_x_y(df)
    for key in fits_to_fit:
        if FIT_IS_COHLESS[key]:
            results['fits'][key] = bootstrap_fit_curves(ts_all, make_bootstrap_fcn(FIT_FCNS[key], 0), nboots)

    for coh, dfc in df.groupby('coherence'):
        ts_cur_coh = as_x_y(dfc)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        results['ntrials'][coh] = len(ts_cur_coh)
        results['fits']['binned_pcor'][coh] = binned_ps(ts_cur_coh, bins, NBOOTS_BINNED_PS, include_se=True)
        for key in fits_to_fit:
            if not FIT_IS_COHLESS[key]:
                results['fits'][key][coh] = bootstrap_fit_curves(ts_cur_coh, make_bootstrap_fcn(FIT_FCNS[key], coh), nboots)
    return results


def fit2(df, bins, fits_to_fit, nboots):
    """
    this uses the pandas dataframe, df, all the way into the fitting function,
        whereas fit() pulls out the df's values and hands the fitting function a big list of data
    fit2 is sooooo much faster. currently it's only available if you're fitting sat-exp
    """
    def fit_inner(df, B, nboots):
        A, T = saturating_exponential.fit_df(df, B)
        logging.info('Fitting.')
        ths = [{'A': A, 'B': B, 'T': T}]
        for inds in bootstrap(df.index.values, nboots):
            A, T = saturating_exponential.fit_df(df.ix[inds,:].copy().reset_index(), B)
            ths.append({'A': A, 'B': B, 'T': T})
        return ths

    results = {}
    results['ntrials'] = {}
    results['cohs'] = sorted(df['coherence'].unique())
    results['fits'] = {'sat-exp': {}}
    results['fits']['binned_pcor'] = {}
    msg = 'sat-exp is assuming duration is binned by duration_index column.'
    logging.warning(msg)
    
    durmap = durmap_fcn(df)
    B = DEFAULT_THETA['sat-exp']['B']
    for coh, dfc in df.groupby('coherence'):
        msg = '{0}%: Found {1} trials'.format(int(coh*100), len(dfc))
        logging.info(msg)
        results['ntrials'][coh] = len(dfc)
        results['fits']['binned_pcor'][coh] = binned_ps(as_x_y(dfc), bins, NBOOTS_BINNED_PS, include_se=True)
        results['fits']['sat-exp'][coh] = fit_inner(dfc, B, nboots)
    return results

def fit_and_write(df, subj, dotmode, fits_to_fit, nboots, bins, outdir, resample=False):
    if resample:
        df = resample_by_grp(df, 5)
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(df), subj, dotmode)
    logging.info(msg)
    if len(fits_to_fit) == 1 and 'sat-exp' in fits_to_fit:
        results = fit2(df, bins, fits_to_fit, nboots)
    else:
        results = fit(df, bins, fits_to_fit, nboots)
    outfile = makefn(outdir, subj, dotmode, '', 'pickle')
    return write_pickle(results, bins, outfile, subj, dotmode)

def parse_outdir(outdir):
    """
    if outdir is just a folder name, assume it belongs in $BASEDIR/res
    """
    if '/' in outdir:
        return os.path.abspath(outdir)
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    return os.path.join(BASEDIR, 'res', outdir)
    
def main(ps, is_agg_subj, fits_to_fit, nboots, outdir, isLongDur):
    df = load(ps, None, 'both' if isLongDur else False)
    bins = list(df.groupby('duration_index', as_index=False).agg(min)['real_duration'].values)
    bins.append(df['real_duration'].max())
    outdir = parse_outdir(outdir)
    res = {}
    for dotmode, dfc in df.groupby('dotmode'):
        if is_agg_subj:
            res[('ALL', dotmode)] = fit_and_write(dfc, AGG_SUBJ_NAME, dotmode, fits_to_fit, nboots, bins, outdir, resample=True)
        else:
            for subj, dfc2 in dfc.groupby('subj'):
                res[(subj, dotmode)] = fit_and_write(dfc2, subj, dotmode, fits_to_fit, nboots, bins, outdir, resample=False)
    write_csv(res, outdir)

if __name__ == '__main__':
    ALL_FITS = FIT_FCNS.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--subj", type=str, help="Restrict fit to a single subject.")
    parser.add_argument('-d', "--dotmode", type=str, help="2D or 3D or (default:) both.")
    parser.add_argument('-a', '--is_agg_subj', action='store_true', default=False)
    parser.add_argument('-f', "--fits", default=ALL_FITS, nargs='*', choices=ALL_FITS, type=str, help="The fitting methods you would like to use, from: {0}".format(ALL_FITS))
    parser.add_argument('-n', "--nboots", default=0, type=int, help="The number of bootstraps of fits")
    parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    args = parser.parse_args()
    
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    main(ps, args.is_agg_subj, args.fits, args.nboots, args.outdir, args.is_long_dur)
