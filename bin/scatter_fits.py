import os
import glob
import pickle
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from scatter_plots import plot_by_key, plot_info
from tools import color_list

DEFAULT_THETA = {'A': 1.0, 'B': 0.5, 'T': 0.001}

def load_inner(infile):
    return pickle.load(open(infile))

def load(indir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INDIR = os.path.join(BASEDIR, 'res', indir)
    fits = {}
    for ind in glob.glob(os.path.join(INDIR, '*.pickle')):
        ind = os.path.abspath(ind)
        fname = os.path.splitext(os.path.split(ind)[-1])[0]
        subj, cond, _ = fname.split('-')
        fits[(subj, cond)] = load_inner(ind)
    return fits

def prep(fits):
    rows = []
    for (subj, cond), fit in fits.iteritems():
        base_row = {'subj': subj, 'dotmode': cond}
        vals = [dict({'coh': coh}.items() + ts[0].items()) for coh, ts in fit['fits']['sat-exp'].iteritems()]
        rows.extend([dict(base_row.items() + val.items()) for val in vals])
    df = pd.DataFrame(rows)
    # df = df.set_index(['subj', 'coh'])
    return df

def plot_all_by_key(df, key):
    y1, y2 = 1000000, -1000000
    cohs = sorted(df['coh'].unique())
    colmap = dict((coh, col) for coh, col in zip([0]*1 + cohs, color_list(len(cohs) + 1, "YlGnBu")))
    for coh in cohs:
        dfc = df[df['coh'] == coh]
        dfc = dfc.set_index('subj')
        ymin, ymax = plot_by_key(dfc, key, 'ALL', label=int(coh*100), color=colmap[coh])
        if ymin < y1:
            y1 = ymin
        if ymax > y2:
            y2 = ymax
    plot_info(key, (y1, y2), True, 'upper left')

def main(indir, key):
    fits = load(indir)
    df = prep(fits)
    plot_all_by_key(df, 'T')
    plt.xlim([None, 200])
    plt.ylim([None, 200])
    plt.show()
    plot_all_by_key(df, 'A')
    plt.xlim([0.4, 1.1])
    plt.ylim([0.4, 1.1])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str)
    parser.add_argument('-k', "--key", default='T', choices=DEFAULT_THETA.keys(), type=str)
    args = parser.parse_args()
    main(args.indir, args.key)
