import os.path
import argparse
from operator import and_, or_ # bitwise-and, e.g. a & b; bitwise-or, e.g. a | b

import numpy as np
import pandas as pd

from pcor_mesh_plot import plot
from sample import rand_inds
from settings import good_subjects, bad_sessions, good_cohs, nsigdots, actualduration, nsigframes
from settings import min_dur, max_dur, NBINS, min_dur_longDur, max_dur_longDur, NBINS_longDur, NBINS_COMBINED

SESSIONS_COLS = [u'index', u'subj', u'dotmode', u'number']
TRIALS_COLS = [u'index', u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct']
COLS = [u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct', u'subj', u'dotmode', u'number']
COL_TYPES = [int, int, float, float, int, float, int, bool, str, str, int]

CURDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
SESSIONS_INFILE = os.path.join(BASEDIR, 'data', 'sessions.csv')
TRIALS_INFILE = os.path.join(BASEDIR, 'data', 'trials.csv')
SESSIONS_INFILE_2 = os.path.join(BASEDIR, 'data', 'sessions-longDur.csv')
TRIALS_INFILE_2 = os.path.join(BASEDIR, 'data', 'trials-longDur.csv')

def load_df(sessions_infile=SESSIONS_INFILE, trials_infile=TRIALS_INFILE):
    df1 = pd.read_csv(sessions_infile, index_col='index')
    assert all(df1.keys() == SESSIONS_COLS[1:])
    df2 = pd.read_csv(trials_infile, index_col='index')
    assert all(df2.keys() == TRIALS_COLS[1:])
    return df2.join(df1, on='session_index')

make_equal_filter = lambda key, val: (key, lambda x: x == val)
make_gt_filter = lambda key, val: (key, lambda x: x >= val)
make_lt_filter = lambda key, val: (key, lambda x: x <= val)

def interpret_filters(args):
    filters = []
    if args is None:
        return filters
    for key, val in args.iteritems():
        if key not in COLS:
            continue
        if val is not None:
            filters.append(make_equal_filter(key, val))
    return filters

def filter_df(df, filters):
    if not filters:
        return df
    preds = []
    for key, pred in filters:
        preds.append(pred(df[key]))
    return df[reduce(and_, preds)]

def shift_bins(df, durbin_3d=2, durbin_floor=1):
    df.ix[df['dotmode'] == '3d', 'duration_index'] = df[df['dotmode'] == '3d']['duration_index'] - durbin_3d
    df = df[df['duration_index'] >= durbin_floor]
    return df

def rebin(df, extraDataset, N=10):
    """
    reassigns duration_index of each trial by rebinning with N log-spaced durations between dur0 and dur1
    removes trials with duration above the dur1
    """
    N += 1
    df['real_duration'] = actualduration(df['duration'])
    if extraDataset == 'longDur':
        dur0 = min_dur_longDur
        dur1 = max_dur_longDur
    elif extraDataset == 'both':
        dur0 = min_dur
        dur1 = max_dur_longDur
    else:
        dur0 = min_dur
        dur1 = max_dur
    dur0 = actualduration(dur0)
    dur1 = actualduration(dur1)
    if extraDataset == 'longDur':
        bins = list(np.linspace(dur0, dur1, N))
    else:
        bins = list(np.logspace(np.log10(dur0), np.log10(dur1), N))
    bins = bins[1:]
    bins[-1] = dur1 + 0.01
    
    bin_lkp = lambda dur: next(i+1 for i, lbin in enumerate(bins + [dur1+1]) if dur < lbin)
    # print df.loc[df['real_duration'] > dur1, :]
    df = df.loc[df['real_duration'] <= dur1, :].copy()
    df.loc[:, 'duration_index'] = df['real_duration'].map(bin_lkp)
    return df

def resample_by_grp(df, mult=5, grp=('dotmode', 'subj')):
    """
    resamples rows of each group in df
    """
    dg = df.groupby(grp, as_index=False)
    ntrials = dg.agg(len)['trial_index']
    N = max(ntrials.min() * mult, ntrials.max())
    return pd.concat([dfc.loc[dfc.index[rand_inds(dfc, N)]] for grp, dfc in dg]).copy()

def default_filter_df(df):
    """
    good_subjects, bad_sessions, good_cohs
    """
    ffs = []
    for dotmode, inds in good_cohs.iteritems():
        pred = (df['dotmode'] == dotmode) & (df['coherence'].isin(inds))
        ffs.append(pred)
    if ffs:
        df = df[reduce(or_, ffs)]
    return df

def filterElbows(df, firstElbow, secondElbow):
    if firstElbow and len(firstElbow) == 2:
        t2d, t3d = firstElbow
        d2d = (df['dotmode'] == '2d') & (df['real_duration'] >= t2d)
        d3d = (df['dotmode'] == '3d') & (df['real_duration'] >= t3d)
        df = df[d2d | d3d]
    if secondElbow and len(secondElbow) == 2:
        t2d, t3d = secondElbow
        d2d = (df['dotmode'] == '2d') & (df['real_duration'] <= t2d)
        d3d = (df['dotmode'] == '3d') & (df['real_duration'] <= t3d)
        df = df[d2d | d3d]
    return df

def load(ps=None, filters=None, extraDataset=None, nbins=None, ignorePastSecondElbow=False, ignoreBeforeFirstElbow=False):
    if extraDataset == 'longDur':
        df = load_df(SESSIONS_INFILE_2, TRIALS_INFILE_2)
        df = rebin(df, extraDataset, NBINS_longDur if nbins is None else nbins)
    elif extraDataset == 'both':
        df = load_df()
        df['isLongDur'] = False
        df2 = load_df(SESSIONS_INFILE_2, TRIALS_INFILE_2)
        df2['isLongDur'] = True
        df = df.append(df2)
        df = rebin(df, extraDataset, NBINS_COMBINED if nbins is None else nbins)
    else:
        df = load_df()
        df = rebin(df, extraDataset, NBINS if nbins is None else nbins)
    fltrs = filters if filters is not None else []
    df = filter_df(df, fltrs + interpret_filters(ps))
    df = filterElbows(df, ignoreBeforeFirstElbow, ignorePastSecondElbow)
    return default_filter_df(df)

def main(ps, isLongDur=False, nbins=None, doPlot=False, outdir=None, outdir_trials=None):
    df = load(ps, None, 'both' if isLongDur else False, nbins)
    if outdir_trials:
        subjs = df['subj'].unique()
        subj = subjs[0] if len(subjs) == 1 else 'ALL'
        df.to_csv(os.path.join(outdir_trials, 'pcor-{0}-trials.csv').format(subj))
    print df.head()
    print df.shape
    if doPlot or outdir:
        df0 = plot(df, doPlot)
        if outdir:
            subjs = df['subj'].unique()
            subj = subjs[0] if len(subjs) == 1 else 'ALL'
            df0.to_csv(os.path.join(outdir, 'pcor-{0}-pts.csv').format(subj))
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for col, typ in zip(COLS, COL_TYPES):
        parser.add_argument("--{0}".format(col), required=False, type=typ, help="{0}".format(col))
    parser.add_argument('-n', '--nbins', required=False, type=int, default=20)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('-o', '--outdir', type=str, default=None, help="outdir for mesh plot")
    parser.add_argument('-g', '--outdir-trials', type=str, default=None, help="outdir for data used")
    args = parser.parse_args()
    main(vars(args), args.is_long_dur, args.nbins, args.plot, args.outdir, args.outdir_trials)
