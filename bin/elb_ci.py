import os.path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

lw = 2
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue']
# rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
rcParams['axes.unicode_minus'] = False
matplotlib.rc('lines', linewidth=lw)

basedir = '../fits'

def load_elb(key, no=''):
    f = 'pcorVsCohByDur_{0}elbow-ALL-params{1}.csv'.format(key, no)
    return pd.read_csv(os.path.join(basedir, f))

lbf = lambda pct: (1.0-pct)/2.0
def elb_ci(df, pct=0.682):
    qs = [lbf(pct), 1-lbf(pct)]
    assert sorted(qs) == qs and len(qs) == 2
    dg = df.groupby('dotmode').quantile(qs + [0.5])
    for k2 in ['2d', '3d']:
        print '--------'
        print k2.upper()
        print '--------'
        for k1 in dg.keys():
            if 'Unnamed' in k1 or 'bi' == k1 or 'm2' == k1:
                continue
            xs = np.array([dg[k1][k2][0.5], dg[k1][k2][qs[0]], dg[k1][k2][qs[1]]])
            if k1 == 'x0' or k1 == 'x1':
                xs = np.exp(xs)
            print '{0}: median = {2:.2f}, {5:.0f}% C.I. = ({3:.2f}, {4:.2f})'.format(k1, k2, xs[0], xs[1], xs[2], 100*pct)

def hists(df, keys=['m1'], nbins=100):
    print 'fits from {0} bootstraps'.format(df['bi'].max())
    fig = plt.figure()
    # ax = plt.gca()
    axs = [fig.add_subplot(2, 1, i+1) for i in range(2)] # for each dotmode
    clrs = ['g','r']
    # bins = np.linspace(df[keys].min().min(), df[keys].max().max(), nbins)
    bins = np.linspace(0.0, 1.0, nbins)
    ymx = 10.0
    for i, (dotmode, dfc) in enumerate(df.groupby('dotmode', as_index=False)):
        for key in keys:
            dfc[key].hist(bins=bins, alpha=1.0, normed=True, color=clrs[i], histtype='stepfilled', lw=2, ax=axs[i])
            # dfc[key].hist(bins=bins, alpha=1.0, normed=True, color=clrs[i], histtype='step', ax=axs[i])
        axs[i].set_xlim(min(bins), max(bins))
        axs[i].set_ylim(0.0, ymx)
        axs[i].grid(False)
        axs[i].plot([0.3, 0.3], [0.0, ymx], '--k', lw=2, label='probability summation')
        axs[i].plot([0.5, 0.5], [0.0, ymx], ':k', lw=2, label='perfect integration')

    leg = axs[-1].legend(loc='lower right', prop={'size': 14})
    leg.get_frame().set_linewidth(1.5)
    axs[-1].invert_yaxis()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

    plt.minorticks_off()
    axs[-1].tick_params(top='off')
    axs[-1].tick_params(right='off')
    axs[-1].tick_params(axis='x', direction='out', width=1.5, length=5)
    axs[-1].tick_params(axis='y', direction='out', width=1.5, length=5)
    axs[-1].patch.set_facecolor('white')
    for axis in ['bottom', 'left']:
        axs[-1].spines[axis].set_linewidth(lw)
    for axis in ['top', 'right']:
        axs[-1].spines[axis].set_linewidth(0)
    axs[0].tick_params(top='off')
    axs[0].tick_params(right='off')
    axs[0].tick_params(axis='x', direction='out', width=1.5, length=5)
    axs[0].tick_params(axis='y', direction='out', width=1.5, length=5)
    axs[0].patch.set_facecolor('white')
    for axis in ['bottom', 'left']:
        axs[0].spines[axis].set_linewidth(lw)
    for axis in ['top', 'right']:
        axs[0].spines[axis].set_linewidth(0)

    plt.show()

if __name__ == '__main__':
    hists(load_elb(''))
    # elb_ci(load_elb(''))
