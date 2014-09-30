import os.path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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
            print '{0}: median = {2:.2f}, {5:.0f}% C.I. = ({3:.2f}, {4:.2f})'.format(k1, k2, dg[k1][k2][0.5], dg[k1][k2][qs[0]], dg[k1][k2][qs[1]], 100*pct)

def load():
    f0 = 'pcorVsCohByDur_thresh-ALL-params.csv'
    f1 = 'pcorVsCohByDur_elbow-ALL-params.csv'
    f2 = 'pcorVsCohByDur_1elbow-ALL-params.csv'
    df0 = pd.read_csv(os.path.join(basedir, f0))
    df1 = pd.read_csv(os.path.join(basedir, f1))
    df2 = pd.read_csv(os.path.join(basedir, f2))
    return df0, df1, df2

font = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'weight': 'bold', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2)

def plot(df0a, df1a, df2a, dotmode, colorB):
    df0 = df0a[df0a['dotmode'] == dotmode]
    df1 = df1a[df1a['dotmode'] == dotmode]
    df2 = df2a[df2a['dotmode'] == dotmode]

    df = df0.groupby('dur', as_index=False)['sens'].agg([np.mean, np.std])['sens'].reset_index()
    if dotmode == '3d':
        df = df[df['dur'] > df['dur'].min()]
        df = df[df['dur'] > df['dur'].min()]

    ptA = df1[['x0','x1','b0','b1','m0','m1']].median()
    df = df[df['dur'] <= np.exp(ptA['x1'])]
    df['y1'] = df['dur']**ptA['m0']*np.exp(ptA['b0'])
    x1 = df['dur'] > np.exp(ptA['x0'])
    df['y1'][x1] = df['dur'][x1]**ptA['m1']*np.exp(ptA['b1'])

    ptB = df2[['b0','m0']].median()
    df['y2'] = df['dur']**ptB['m0']*np.exp(ptB['b0'])

    colorA = '0.7'
    sz = 50
    lw = 2
    # colorB = [0.0, 0.7, 0.0]
    df['y1e'] = (df['y1'] - df['mean'])/df['std']
    df['y2e'] = (df['y2'] - df['mean'])/df['std']
    plt.plot(df['dur'], df['y2e'], lw=lw, c=colorB, label='1-elb', zorder=4)
    plt.plot(df['dur'], df['y1e'], lw=lw, c=colorA, label='2-elb', zorder=4)
    plt.gca().fill_between(df['dur'], 0.0, df['y2e'], facecolor=colorB, alpha=0.6, zorder=3)
    plt.gca().fill_between(df['dur'], 0.0, df['y1e'], facecolor=colorA, alpha=0.6, zorder=3)
    plt.scatter(df['dur'], df['y2e'], sz, c=colorB, zorder=5)
    plt.scatter(df['dur'], df['y1e'], sz, c=colorA, zorder=5)

    plt.plot([np.exp(ptA['x0']), np.exp(ptA['x0'])], [-10, 10], '--', lw=lw, c=colorA, zorder=1)
    plt.plot([df['dur'].min(), df['dur'].max() + 50], [0, 0], '--', lw=lw, c='gray', zorder=1)

    # plt.plot(df['dur'], df['mean'])
    # plt.scatter(df['dur'], df['y1'], c='g')
    # plt.scatter(df['dur'], df['y2'], c='r')
    # plt.yscale('log')
    ys = np.hstack([df['y1e'].values, df['y2e'].values])
    plt.xscale('log')
    plt.xlim(df['dur'].min()-2, df['dur'].max()+30)
    plt.ylim(ys.min()-1, ys.max()+1)
    plt.xlabel('duration (msec)')
    plt.ylabel('motion sensitivity')
    leg = plt.legend()
    leg.get_frame().set_linewidth(1.5)
    plt.xticks([50, 100, 500], [50, 100, 500])
    plt.minorticks_off()
    plt.gca().tick_params(top='off')
    plt.gca().tick_params(right='off')
    plt.gca().tick_params(axis='x', direction='out', width=1.5, length=5)
    plt.gca().tick_params(axis='y', direction='out', width=1.5, length=5)
    plt.gcf().patch.set_facecolor('white')
    for axis in ['bottom', 'left']:
        plt.gca().spines[axis].set_linewidth(lw)
    for axis in ['top', 'right']:
        plt.gca().spines[axis].set_linewidth(0)
    plt.show()

def main():
    df0, df1, df2 = load()
    plot(df0, df1, df2, '2d', [0.0, 0.7, 0.0])
    # plot(df0, df1, df2, '3d', [0.7, 0.0, 0.0])

if __name__ == '__main__':
    elb_ci(load_elb(''))
    # main()
