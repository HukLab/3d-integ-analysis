import glob
import os.path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import color_list

CURDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))

indir_fcn = lambda indir: os.path.join(BASEDIR, 'plots', indir)

def load(indir):
    df = pd.DataFrame()
    for ind in glob.glob(os.path.join(indir_fcn(indir), 'pcorVsCohByDur_elbow-*-params.csv')):
        ind = os.path.abspath(ind)
        fname = os.path.splitext(os.path.split(ind)[-1])[0]
        subj = fname.split('pcorVsCohByDur_elbow-')[1].split('-params')[0].lower()
        df0 = pd.read_csv(ind)
        df0['subj'] = subj
        df = df.append(df0)
    return df

def make_cmap(df):
    subjs = df['subj'].unique()
    colors = color_list(len(subjs) + 2)
    return dict(zip(subjs, colors))

def main(indir, key1, key2=None):
    df = load(indir)
    cmap = make_cmap(df)
    cmap['all'] = 'gray'
    if not key2:
        dfm = df.groupby(['subj', 'dotmode'], as_index=False).agg(np.median)[['subj', 'dotmode', key1]]
        # dfm = df[df['subj'] == 'lnk']
        for subj, ry in dfm.groupby('subj'):
            # x = ry[ry.dotmode == '2d'][key1].values
            # y = ry[ry.dotmode == '3d'][key1].values
            x, y = ry.sort('dotmode')[key1].values
            plt.scatter(x, y, facecolors=cmap[subj], edgecolors='none', s=80)
            plt.text(x, y, '{0}'.format(subj))
        m1, m2 = dfm[key1].min(), dfm[key1].max()
        plt.plot([m1, m2], [m1, m2], 'k--')
        plt.xlabel('2d')
        plt.ylabel('3d')
    else:
        dfm = df.groupby(['subj', 'dotmode'], as_index=False).agg(np.median)[['subj', 'dotmode', key1, key2]]
        for (subj, dotmode), r in dfm.groupby(['subj', 'dotmode']):
            filled = 'k' if dotmode == '3d' else cmap[subj]
            plt.scatter(r[key1], r[key2], facecolors=cmap[subj], edgecolors=filled, s=80)
            # plt.text(r[key1], r[key2], '{0}'.format(dotmode.upper()))
        plt.xlabel(key1)
        plt.ylabel(key2)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str)
    parser.add_argument('-x', "--key1", required=True, type=str)
    parser.add_argument('-y', "--key2", required=False, default=None, type=str)
    args = parser.parse_args()
    main(args.indir, args.key1, args.key2)
