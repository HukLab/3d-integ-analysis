import os.path
import argparse
from operator import and_ # bitwise-and, e.g. a & b

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

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

def plot_surface(xs, ys, zs, resX=50, resY=50):
    xi = np.linspace(min(xs), max(xs), resX)
    yi = np.linspace(min(ys), max(ys), resY)
    Z = griddata(xs, ys, zs, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def save_or_show(show, outfile):
    if not show:
        return
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

def plot(df0, show=True, outfile=None, fig=None, color='c'):
    if fig is None:
        fig = plt.figure()
    if len(set(df0['dotmode'])) == 2:
        plot(df0[df0['dotmode']=='2d'], False, None, fig, 'g')
        plot(df0[df0['dotmode']=='3d'], False, None, fig, 'r')
        save_or_show(show, outfile)
        return
    df2 = df0[['coherence', 'duration_index', 'correct']]
    df = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)
    data = zip(*df.values)
    if len(df.keys()) == 3:
        ax = fig.gca(projection='3d')
        x,y,z = data
        x = np.log(100.*np.array(x))
        ax.scatter(x,y,z, color=color)
        ax.plot_wireframe(*plot_surface(x,y,z), color=color)
        ax.set_xlabel('log(coherence)')
        ax.set_ylabel('duration bin')
        ax.set_zlabel('p correct')
        ax.set_zlim([0.0, 1.0])
    elif len(df.keys()) == 2:
        ax = fig.gca(projection='2d')
        ax.scatter(*data, color=color)
    else:
        raise Exception("too many dimensions in d: {0}".format(len(df.keys())))
    # ax.legend()
    save_or_show(show, outfile)

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
