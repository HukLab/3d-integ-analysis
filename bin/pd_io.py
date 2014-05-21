import os.path
import argparse
from operator import and_ # bitwise-and, e.g. a & b

import pandas as pd

from pd_plot import plot

COLS = [u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct', u'subj', u'dotmode', u'number']
COL_TYPES = [int, int, float, float, int, float, int, bool, str, str, int]

CURDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
PLOT_OUTFILE = os.path.join(BASEDIR, 'res', 'misc', 'tmp.png')
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
    filters.append(make_gt_filter('coherence', 0.0))
    for key, val in args.iteritems():
        assert key in COLS
        if val is not None:
            filters.append(make_equal_filter(key, val))
    return filters

def filter(df, filters):
    preds = []
    for key, pred in filters:
        preds.append(pred(df[key]))
    return df[reduce(and_, preds)]

def main(args):
    df = load()
    print df.shape
    df = filter(df, interpret_filters(args))
    print df.head()
    print df.shape
    plot(df)

parser = argparse.ArgumentParser()
for col, typ in zip(COLS, COL_TYPES):
    parser.add_argument("--{0}".format(col), required=False, type=typ, help="{0}".format(col))
args = parser.parse_args()
if __name__ == '__main__':
    main(vars(args))
