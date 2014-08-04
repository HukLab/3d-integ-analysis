import json
import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from tools import color_list
import saturating_exponential
from settings import DEFAULT_THETA, min_dur, max_dur

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
        plot_inner(ax, df)
        plot_info(ax, title)
        plt.show()

def write_json((xsp, ysp), (xs, ys, th), subj, dotmode, outdir):
    outfile = os.path.join(outdir, 'pcorVsDurByCoh-{subj}-{dotmode}.json'.format(subj=subj, dotmode=dotmode))
    with open(outfile, 'w') as f:
        obj = {'binned': {'xs': list(xsp), 'ys': list(ysp)}, 'fit': {'xs': list(xs), 'ys': list(ys), 'theta': list(th)}}
        json.dump(obj, f)

def fit_curve(df, A=None, T=None):
    B = DEFAULT_THETA['sat-exp']['B']
    if A is None and T is None:
        data = df[['duration', 'correct']].values
        th = saturating_exponential.fit(data, (None, B, None), quick=True)
        if len(th) > 0 and th[0]['success']:
            A, T = th[0]['x']
        else:
            return None, None, None
    xs = np.logspace(np.log10(min_dur), np.log10(max_dur))
    ys = saturating_exponential.saturating_exp(xs, A, B, T)
    sec_to_ms = lambda xs: [x*1000 for x in xs]
    return sec_to_ms(xs), ys, (A, B, T)

def fit(args, outdir, isLongDur=False):
    df = load(args, None, isLongDur)
    subj = subj_label(df)
    durmap = durmap_fcn(df)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        isp, ysp = zip(*df_dotmode.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [1000*durmap[i] for i in isp]
        xs, ys, th = fit_curve(df_dotmode)
        th = list(th)
        th[-1] += 30 # for delay
        if outdir:
            write_json((xsp, ysp), (xs, ys, th), subj, dotmode, outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--fit', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--is-long-dur', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    if args.fit:
        fit(ps, args.outdir, args.is_long_dur)
    else:
        plot(ps, args.is_long_dur)
