import json
import os.path
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pd_io import load,  resample_by_grp
from tools import color_list
import saturating_exponential
from settings import DEFAULT_THETA, min_dur, max_dur, min_dur_longDur, max_dur_longDur

durmap_fcn = lambda df: dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
colmap_fcn = lambda cohs, name="YlGnBu": dict((coh, col) for coh, col in zip([0]*2 + cohs, color_list(len(cohs) + 2, name)))

def subj_label(df, default='ALL'):
    subjs = df['subj'].unique()
    return subjs[0].upper() if len(subjs) == 1 else default

def plot_inner(ax, df):
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = sorted(df['coherence'].unique())
    colmap = colmap_fcn(cohs)
    for coh, df_coh in df.groupby('coherence'):
        isp, ysp = zip(*df_coh.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [durmap[i]*1000 for i in isp]
        ax.plot(xsp, ysp, color=colmap[coh], label="%0.2f" % coh, marker='o', linestyle='-')

def plot_info(ax, title):
    plt.title('{0}: % correct vs. duration'.format(title))
    plt.xlabel('duration')
    plt.ylabel('% correct')
    # plt.xscale('log')
    # plt.xlim([30, 1050])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot(args, isLongDur=False):
    df = load(args, None, isLongDur)
    subj = subj_label(df, '')
    for dotmode, df_dotmode in df.groupby('dotmode'):
        fig = plt.figure()
        ax = plt.subplot(111)
        title = '{0}, {1}'.format(dotmode, subj)
        plot_inner(ax, df_dotmode)
        plot_info(ax, title)
        plt.show()

def prep_res((xsp, ysp, zsp), (xs, ys, th)):
    """
    SA1 = ['subj', 'dotmode', 'is_bin_or_fit', 'x', 'y', 'ntrials']
    SA2 = ['subj', 'dotmode', 'A', 'B', 'T']
    """
    df1 = pd.DataFrame({'xs': xsp, 'ys': ysp, 'ntrials': zsp})
    df1['is_bin_or_fit'] = 'bin'
    dft = pd.DataFrame({'xs': xs, 'ys': ys})
    dft['is_bin_or_fit'] = 'fit'
    df1 = df1.append(dft)
    df2 = pd.DataFrame([{'A': th[0], 'B': th[1], 'T': th[2]}])
    return df1, df2

def finish_res(res, subj):
    d1, d2 = None, None
    for (dm, coh), (df1, df2) in res.iteritems():
        df1['dotmode'] = dm
        df2['dotmode'] = dm
        if coh is not None:
            df1['coh'] = coh
            df2['coh'] = coh
        d1 = d1.append(df1) if d1 is not None else df1
        d2 = d2.append(df2) if d2 is not None else df2
    d1['subj'] = subj
    d2['subj'] = subj
    return d1, d2

def write_csv(df1, df2, subj, collapseCoh, outdir):
    if collapseCoh:
        key = lambda kind: 'pcorVsDurByCoh-{subj}-{kind}.csv'.format(kind=kind, subj=subj)
    else:
        key = lambda kind: 'fitCurveVsDurByCoh-{subj}-{kind}.csv'.format(kind=kind, subj=subj)
    outfile_fcn = lambda kind: os.path.join(outdir, key(kind))
    df1.to_csv(outfile_fcn('pts'))
    df2.to_csv(outfile_fcn('params'))

def plot_fit(df1, df2, collapseCoh):
    def inner_fit(df, key, colmap):
        for grp, dfc in df[df['is_bin_or_fit'] == 'bin'].groupby(key):
            dfc.plot('xs', 'ys', kind='scatter', ax=plt.gca(), color=colmap[grp])
        for grp, dfc in df[df['is_bin_or_fit'] == 'fit'].groupby(key):
            dfc.plot('xs', 'ys', ax=plt.gca(), color=colmap[grp])
        plt.xscale('log')
        plt.ylim([0.4, 1.0])
        plt.show()

    if collapseCoh:
        colmap = {'2d': 'g', '3d': 'r'}
        inner_fit(df1, 'dotmode', colmap)
    else:
        cohs = sorted(df1['coh'].unique())
        colmap = {}
        for (dotmode, name) in [('2d', 'Greens'), ('3d', 'Reds')]:
            items = [(dotmode, coh) for coh in cohs]
            colmap.update(colmap_fcn(items, name))
        for dm, dfc in df1.groupby('dotmode'):
            inner_fit(dfc, ['dotmode', 'coh'], colmap)

def fit_curve(df, (dur0, dur1)):
    B = DEFAULT_THETA['sat-exp']['B']
    A, T = saturating_exponential.fit_df(df, B)
    if A is None:
        return None, None, None
    xs = df.sort('real_duration')['real_duration'].unique() #np.logspace(np.log10(dur0), np.log10(dur1))
    ys = saturating_exponential.saturating_exp(xs, A, B, T)
    sec_to_ms = lambda xs: [x*1000 for x in xs]
    return sec_to_ms(xs), ys, (A, B, T)

def make_data(df, durmap):
    dfc1 = df.groupby('duration_index', as_index=False)['correct'].agg([np.mean, len])['correct'].reset_index()
    xs, ys, zs = zip(*dfc1[['duration_index','mean','len']].values)
    xs = np.array([1000.0*durmap[i] for i in xs])
    ys = np.array(ys).astype('float')
    zs = np.array(zs)
    return xs, ys, zs

def fit_df(df, res, grp, dur_rng, durmap):
    xs, ys, th = fit_curve(df, dur_rng)
    if not th:
        return (None, None), (None, None, None)
    th = list(th)
    # th[-1] += 30 # for delay
    if not th:
        print 'ERROR: No fits found.'
    print grp, th

    # isp, ysp = zip(*df.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
    # xsp = [1000*durmap[i] for i in isp]
    xsp, ysp, zsp = make_data(df, durmap)
    res[grp] = prep_res((xsp, ysp, zsp), (xs, ys, th))
    return res

def fit(args, outdir, collapseCoh=False, isLongDur=False, resample=None, plot=False):
    df = load(args, None, 'both' if isLongDur else False)
    if resample:
        df = resample_by_grp(df, resample)
    dur_rng = (min_dur, max_dur_longDur) if isLongDur else (min_dur, max_dur)
    subj = subj_label(df)
    durmap = durmap_fcn(df)
    res = {}
    for grp, df_dotmode in df.groupby('dotmode' if collapseCoh else ['dotmode', 'coherence']):
        key = grp if not collapseCoh else (grp, None)
        res = fit_df(df_dotmode, res, key, dur_rng, durmap)
    df1, df2 = finish_res(res, subj)
    if outdir:
        write_csv(df1, df2, subj, collapseCoh, outdir)
    if plot:
        plot_fit(df1, df2, collapseCoh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--fit', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    parser.add_argument('--collapse-coh', action='store_true', default=False)
    parser.add_argument('-r', '--resample', type=int, default=0)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    if args.fit:
        fit(ps, args.outdir, args.collapse_coh, args.is_long_dur, args.resample, args.plot)
    else:
        plot(ps, args.is_long_dur)
