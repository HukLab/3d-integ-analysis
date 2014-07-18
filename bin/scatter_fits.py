import os
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scatter_plots import plot_by_key, plot_info
from settings import DEFAULT_THETA
from tools import color_list

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

def prep(fits, fit):
    rows = []
    for (subj, cond), res in fits.iteritems():
        base_row = {'subj': subj, 'dotmode': cond}
        vals = [dict({'coh': coh}.items() + ts[0].items()) for coh, ts in res['fits'][fit].iteritems()]
        rows.extend([dict(base_row.items() + val.items()) for val in vals])
    df = pd.DataFrame(rows)
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
    mean_x, mean_y = df.groupby('dotmode').agg(np.mean)[key]
    plt.plot(mean_x, mean_y, label='mean', marker='o', linestyle='', color='red')
    plt.annotate('average {2} = {0:.2f}, {1:.2f}'.format(mean_x, mean_y, key), xy=(mean_x, mean_y), xytext=(mean_x*1.07, mean_y))
    plot_info(key, (y1, y2), True, 'upper left')

def main(indir, fit, key):
    fits = load(indir)
    df = prep(fits, fit)
    # df = df[df[key] <= 200]
    # df = df[df['coh'] != 0.03]
    plot_all_by_key(df, key)
    # plt.xlim([None, 200])
    # plt.ylim([None, 200])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str)
    parser.add_argument('-f', "--fit", required=True, choices=DEFAULT_THETA.keys(), type=str)
    parser.add_argument('-k', "--key", required=True, type=str)
    args = parser.parse_args()
    main(args.indir, args.fit, args.key)
