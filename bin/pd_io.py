import os.path
import argparse
from operator import and_, or_ # bitwise-and, e.g. a & b; bitwise-or, e.g. a | b

import pandas as pd

from pd_plot import plot
from session_info import good_subjects, bad_sessions, good_cohs

COLS = [u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct', u'subj', u'dotmode', u'number']
COL_TYPES = [int, int, float, float, int, float, int, bool, str, str, int]

CURDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
SESSIONS_INFILE = os.path.join(BASEDIR, 'csv', 'sessions.csv')
TRIALS_INFILE = os.path.join(BASEDIR, 'csv', 'trials.csv')

def load(sessions_infile=SESSIONS_INFILE, trials_infile=TRIALS_INFILE):
    df1 = pd.read_csv(sessions_infile, index_col='index')
    assert all(df1.keys() == [u'subj', u'dotmode', u'number'])
    df2 = pd.read_csv(trials_infile, index_col='index')
    assert all(df2.keys() == [u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct'])
    return df2.join(df1, on='session_index')

make_equal_filter = lambda key, val: (key, lambda x: x == val)
make_gt_filter = lambda key, val: (key, lambda x: x > val)

def interpret_filters(args):
    filters = []
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
            pred = (df['subj'] == subj) & (df['dotmode'] == dotmode) & ~(df['session_index'].isin(inds))
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

def plot_fit_error(df, indir, subj, cond, fit):
    import numpy as np
    import matplotlib.pyplot as plt
    from pd_plot import plot_inner
    from plotCurveVsDurByCoh import load_pickle
    from drift_diffuse import drift_diffusion
    from saturating_exponential import saturating_exp

    fig = plt.figure()

    # observed data
    df2 = df[['coherence', 'duration_index', 'correct']]
    df3 = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)
    # plot_inner(zip(*df3.values), fig, 'c')

    # fit data
    indir = os.path.join(BASEDIR, 'res', indir)
    res = load_pickle(indir, subj, cond)['fits'][fit]
    if len(res) == 1:
        fnd = True
        if fit == 'drift':
            X0, K = res[0]['X0'], res[0]['K']
        elif fit == 'sat-exp':
            A, B, T = res[0]['A'], res[0]['B'], res[0]['T']
    else:
        fnd = False
        res = dict((float(r), d) for r,d in res.iteritems())
    durs = dict(np.fliplr(df[['duration', 'duration_index']].groupby('duration_index', as_index=False).aggregate(np.min).values))
    cohs = df.coherence.unique()
    vals = []
    for c in sorted(cohs):
        for t in sorted(durs):
            if not fnd:
                if fit == 'drift':
                    X0, K = res[c][0]['X0'], res[c][0]['K']
                elif fit == 'sat-exp':
                    A, B, T = res[c][0]['A'], res[c][0]['B'], res[c][0]['T']
            if fit == 'drift':
                val = drift_diffusion((c, t), K, X0)
            elif fit == 'sat-exp':
                val = saturating_exp(t, A, B, T)
            val2 = float(df3[(df3.coherence==c) & (df3.duration_index==durs[t])].correct)
            # vals.append((c, durs[t], val))
            vals.append((c, durs[t], val2-val))
    plot_inner(zip(*vals), fig, 'b')

    plt.show()

def main(args):
    df = load()
    print df.shape
    df = filter_df(df, interpret_filters(args))
    df = default_filter_df(df)
    print df.head()
    print df.shape
    plot(df)

    # plot_fit_error(df, 'sat-exp', 'ALL', args['dotmode'], 'sat-exp')
    # plot_fit_error(df, 'drift-fits-free', 'ALL', args['dotmode'], 'drift')
    # plot_fit_error(df, 'drift-fits-og', 'ALL', args['dotmode'], 'drift')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for col, typ in zip(COLS, COL_TYPES):
        parser.add_argument("--{0}".format(col), required=False, type=typ, help="{0}".format(col))
    args = parser.parse_args()
    main(vars(args))
