import os.path
import argparse
from operator import and_, or_ # bitwise-and, e.g. a & b; bitwise-or, e.g. a | b

import numpy as np
import pandas as pd

from pcor_mesh_plot import plot
from sample import rand_inds
from settings import good_subjects, bad_sessions, good_cohs, nsigdots, actualduration, min_dur, max_dur, NBINS, min_dur_longDur, max_dur_longDur, NBINS_longDur, nsigframes

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

def interpret_filters(args):
    filters = []
    if args is None:
        return filters
    for key, val in args.iteritems():
        assert key in COLS
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

def rebin(df, isLongDur, N=10):
    """
    reassigns duration_index of each trial by rebinning with N log-spaced durations between dur0 and dur1
    removes trials with duration above the dur1
    """
    N += 1
    df['real_duration'] = actualduration(df['duration'])
    if isLongDur:
        dur0 = min_dur_longDur
        dur1 = max_dur_longDur
    else:
        dur0 = min_dur
        dur1 = max_dur
    dur0 = actualduration(dur0)
    dur1 = actualduration(dur1)
    if isLongDur:
        bins = list(np.linspace(dur0, dur1, N))
    else:
        bins = list(np.logspace(np.log10(dur0), np.log10(dur1), N))
    bins = bins[1:]
    bins[-1] = dur1 + 0.01
    
    if True:
        """
        as per leor: first 5 frames should be their own bin; after that, binned as per normal
        so need to combine nf and bin_lkp somehow
        """
        max_nframe_as_bin = 5
        nf = sorted(nsigframes(df['real_duration']).unique().tolist())
        bin_lkp = lambda dur: next(i+1 for i, lbin in enumerate(bins + [dur1+1]) if dur < lbin)

        bins1 = df['real_duration'].unique().tolist()[:nf.index(max_nframe_as_bin+1)]
        bins = bins1 + [b for b in bins if b > bins1[-1]]
        bins = bins[1:]

    df = df.loc[df['real_duration'] <= dur1, :].copy()
    df.loc[:, 'duration_index'] = df['real_duration'].map(bin_lkp)
    return df

def resample_by_grp(df, mult, grp=('dotmode', 'subj')):
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
    # good_subjects
    ffs = []
    for dotmode, subjs in good_subjects.iteritems():
        pred = (df['dotmode'] == dotmode) & (df['subj'].isin(subjs))
        ffs.append(pred)
    if ffs:
        df = df[reduce(or_, ffs)]
    # bad_sessions
    ffs = []
    for dotmode, lkp in bad_sessions.iteritems():
        for subj, inds in lkp.iteritems():
            pred = (df['subj'] == subj) & (df['dotmode'] == dotmode) & ~(df['number'].isin(inds))
            ffs.append(pred)
    if ffs:
        df = df[reduce(or_, ffs)]
    # good_cohs
    ffs = []
    for dotmode, inds in good_cohs.iteritems():
        pred = (df['dotmode'] == dotmode) & (df['coherence'].isin(inds))
        ffs.append(pred)
    if ffs:
        df = df[reduce(or_, ffs)]
    return df

def load(args=None, filters=None, isLongDur=False):
    if isLongDur:
        df = load_df(SESSIONS_INFILE_2, TRIALS_INFILE_2)
    else:
        df = load_df()
    df = rebin(df, isLongDur, NBINS_longDur if isLongDur else NBINS)
    fltrs = filters if filters is not None else []
    df = filter_df(df, fltrs + interpret_filters(args))
    return default_filter_df(df) if not isLongDur else df

def main(args, isLongDur=False):
    df = load(args, None, isLongDur)
    print df.head()
    print df.shape
    plot(df)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for col, typ in zip(COLS, COL_TYPES):
        parser.add_argument("--{0}".format(col), required=False, type=typ, help="{0}".format(col))
    args = parser.parse_args()
    main(vars(args))
