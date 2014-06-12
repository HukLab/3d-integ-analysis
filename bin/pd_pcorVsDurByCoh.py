import argparse
import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from tools import color_list
import saturating_exponential
from session_info import DEFAULT_THETA, min_dur, max_dur

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

def plot_inner(ax, df):
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = sorted(df['coherence'].unique())
    colmap = dict((coh, col) for coh, col in zip([0]*2 + cohs, color_list(len(cohs) + 2, "YlGnBu")))
    for coh, df_coh in df.groupby('coherence'):
        isp, ysp = zip(*df_coh.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [durmap[i]*1000 for i in isp]
        ax.plot(xsp, ysp, color=colmap[coh], label="%0.2f" % coh, marker='o', linestyle='-')

def plot_inner_across_cohs(ax, df, dotmode, fit=True):
    color='g' if dotmode == '2d' else 'r'
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    isp, ysp = zip(*df.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
    xsp = [durmap[i]*1000 for i in isp]
    ax.plot(xsp, ysp, color=color, label=dotmode, marker='o', linestyle='')
    if fit:
        if dotmode == '2d':
            A, T = 0.8691109, 63.54061409
        elif dotmode == '3d': # 0.84060266  496.6706102
            A, T = 0.7330778, 135.2477195
        xs, ys, th = fit_curve(df)#, A, T)
        th = list(th)
        th[-1] += 39 # for delay
        if xs is not None:
            ax.plot(xs, ys, color=color)
            plt.axvline(th[-1], color=color, linestyle='--')
            plt.text(th[-1] + 5, 0.5 if dotmode == '2d' else 0.45, 'tau={0:.2f}'.format(th[-1]), color=color)
            # plt.axhline(th[0], color=color, linestyle='dotted')

def plot_info(ax, title):
    plt.title('{0}: % correct vs. duration'.format(title))
    plt.xlabel('duration')
    plt.ylabel('% correct')
    plt.xscale('log')
    plt.xlim([30, 1050])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot(ax, df, dotmode, show_for_each_dotmode):
    subjs = df['subj'].unique()
    if show_for_each_dotmode:
        plot_inner(ax, df)
        plot_info(ax, dotmode + ', {0}'.format(subjs[0].upper() if len(subjs) == 1 else ''))
    else:
        plot_inner_across_cohs(ax, df, dotmode)

def main(args, show_for_each_dotmode, savefig=False):
    df = load(args)
    subjs = df['subj'].unique()
    outfile = '/Users/mobeets/Desktop/plots/{subj}-{dotmode}.png'
    if not show_for_each_dotmode:
        fig = plt.figure()
        ax = plt.subplot(111)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        if show_for_each_dotmode:
            fig = plt.figure()
            ax = plt.subplot(111)
        plot(ax, df_dotmode, dotmode, show_for_each_dotmode)
        if show_for_each_dotmode:
            if savefig:
                plt.savefig(outfile.format(subj=subjs[0] if len(subjs) == 1 else 'ALL', dotmode=dotmode))
            else:
                plt.show()
    if not show_for_each_dotmode:
        plot_info(ax, 'all cohs' + ', {0}'.format(subjs[0].upper() if len(subjs) == 1 else ''))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--join-dotmode', action='store_true', default=False)
    parser.add_argument('--savefig', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    main(ps, not args.join_dotmode, args.savefig)
