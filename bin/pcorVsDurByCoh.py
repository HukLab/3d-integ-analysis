import json
import os.path
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pd_io import load
from tools import color_list
import saturating_exponential
from settings import DEFAULT_THETA, min_dur, max_dur, min_dur_longDur, max_dur_longDur

durmap_fcn = lambda df: dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)

def subj_label(df, default='ALL'):
    subjs = df['subj'].unique()
    return subjs[0].upper() if len(subjs) == 1 else default

def plot_inner(ax, df):
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = sorted(df['coherence'].unique())
    colmap = dict((coh, col) for coh, col in zip([0]*2 + cohs, color_list(len(cohs) + 2, "YlGnBu")))
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

# def write_json((xsp, ysp), (xs, ys, th), subj, dotmode, outdir):
#     outfile = os.path.join(outdir, 'pcorVsDurByCoh-{subj}-{dotmode}.json'.format(subj=subj, dotmode=dotmode))
#     with open(outfile, 'w') as f:
#         obj = {'binned': {'xs': list(xsp), 'ys': list(ysp)}, 'fit': {'xs': list(xs), 'ys': list(ys), 'theta': list(th)}}
#         json.dump(obj, f)

def prep_csv((xsp, ysp), (xs, ys, th)):
    """
    SA1 = ['subj', 'dotmode', 'is_bin_or_fit', 'x', 'y']
    SA2 = ['subj', 'dotmode', 'A', 'B', 'T']
    """
    df1 = pd.DataFrame({'xs': xsp, 'ys': ysp})
    df1['is_bin_or_fit'] = 'bin'
    dft = pd.DataFrame({'xs': xs, 'ys': ys})
    dft['is_bin_or_fit'] = 'fit'
    df1 = df1.append(dft)
    df2 = pd.DataFrame({'theta': th})
    return df1, df2

def write_csv(res, subj, outdir):
    d1, d2 = None, None
    for dm, (df1, df2) in res.iteritems():
        df1['dotmode'] = dm
        df2['dotmode'] = dm
        d1 = d1.append(df1) if d1 is not None else df1
        d2 = d2.append(df2) if d2 is not None else df2
    d1['subj'] = subj
    d2['subj'] = subj
    outfile_fcn = lambda kind: os.path.join(outdir, 'pcorVsDurByCoh-{subj}-{kind}.csv'.format(kind=kind, subj=subj))
    d1.to_csv(outfile_fcn('pts'))
    d2.to_csv(outfile_fcn('params'))

def fit_curve(df, (dur0, dur1)):
    B = DEFAULT_THETA['sat-exp']['B']
    A, T = saturating_exponential.fit_df(df, B)
    if A is None:
        return None, None, None
    xs = df.sort('real_duration')['real_duration'].unique() #np.logspace(np.log10(dur0), np.log10(dur1))
    ys = saturating_exponential.saturating_exp(xs, A, B, T)
    sec_to_ms = lambda xs: [x*1000 for x in xs]
    return sec_to_ms(xs), ys, (A, B, T)

def fit(args, outdir, isLongDur=False, plot=True):
    df = load(args, None, 'both' if isLongDur else False)
    dur_rng = (min_dur, max_dur_longDur) if isLongDur else (min_dur, max_dur)
    subj = subj_label(df)
    durmap = durmap_fcn(df)
    res = {}
    for dotmode, df_dotmode in df.groupby('dotmode'):
        xs, ys, th = fit_curve(df_dotmode, dur_rng)
        if not th:
            print 'ERROR: No fits found.'
            continue
        th = list(th)
        # th[-1] += 30 # for delay
        print dotmode, th

        isp, ysp = zip(*df_dotmode.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [1000*durmap[i] for i in isp]
        if plot:
            plt.scatter(xsp, ysp, color='g' if dotmode == '2d' else 'r')
            plt.plot(xs, ys, color='g' if dotmode == '2d' else 'r')
        if outdir:
            res[dotmode] = prep_csv((xsp, ysp), (xs, ys, th))
    write_csv(res, subj, outdir)
    if plot:
        # plt.xscale('log')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--fit', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    if args.fit:
        fit(ps, args.outdir, args.is_long_dur)
    else:
        plot(ps, args.is_long_dur)
