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

def main(args):
    df = load()
    print df.shape
    df = filter_df(df, interpret_filters(args))
    df = default_filter_df(df)
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
