import glob
import os.path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import color_list

def load(indir):
    df = pd.DataFrame()
    for ind in glob.glob(os.path.join(os.path.abspath(indir), 'pcorVsCohByDur_elbow-*-params.csv')):
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

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def main(indir, key1, key2=None):
    from matplotlib.patches import Rectangle
    df = load(indir)
    cmap = make_cmap(df)
    cmap['all'] = 'k'
    if not key2:
        dfm = df.groupby(['subj', 'dotmode'], as_index=False).agg(np.median)[['subj', 'dotmode', key1]]
        # df1 = df.groupby(['subj', 'dotmode'], as_index=False).agg([percentile(25), percentile(50), percentile(75)])
        # dfm = df1[key1].reset_index()

        # 1/0
        # dfm = df[df['subj'] == 'lnk']
        for subj, ry in df.groupby('subj'):
            # x0, y0 = ry.sort('dotmode')['percentile_25'].values
            # x1, y1 = ry.sort('dotmode')['percentile_75'].values
            # plt.gca().add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="none", edgecolor=cmap[subj], lw=2))

            # x, y = ry.sort('dotmode')['percentile_50'].values
            
            # x, y = ry.sort('dotmode')[key1].values
            x = ry[ry.dotmode == '2d'][key1].values
            y = ry[ry.dotmode == '3d'][key1].values

            plt.scatter(x, y, facecolors=cmap[subj], edgecolors='none', s=80)
            plt.text(x, y, '{0}'.format(subj))
        # m1, m2 = dfm['percentile_50'].min(), dfm['percentile_50'].max()
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
    """
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj --subj krm
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj --subj lnk
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj --subj huk
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj --subj lkc
    python pmf_fit.py -l -e 2 --enforce-zero -n 20 -b 1000 --outdir ../plots/by-subj --subj klb
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str)
    parser.add_argument('-x', "--key1", required=True, type=str)
    parser.add_argument('-y', "--key2", required=False, default=None, type=str)
    args = parser.parse_args()
    main(args.indir, args.key1, args.key2)
