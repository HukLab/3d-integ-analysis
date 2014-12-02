import os.path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

basedir = '../fits'

def load():
    f0 = 'pcorVsCohByDur_thresh-ALL-params.csv'
    f1 = 'pcorVsCohByDur_elbow-ALL-params.csv'
    f2 = 'pcorVsCohByDur_0elbow-ALL-params.csv'
    df0 = pd.read_csv(os.path.join(basedir, f0))
    df1 = pd.read_csv(os.path.join(basedir, f1))
    df2 = pd.read_csv(os.path.join(basedir, f2))
    return df0, df1, df2

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue']
# rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
rcParams['axes.unicode_minus'] = False
matplotlib.rc('lines', linewidth=2)

def plot(df0a, df1a, df2a, dotmode, colorA, colorB):
    df0 = df0a[df0a['dotmode'] == dotmode]
    df1 = df1a[df1a['dotmode'] == dotmode]
    df2 = df2a[df2a['dotmode'] == dotmode]

    df = df0.groupby('dur', as_index=False)['sens'].agg([np.mean, np.std])['sens'].reset_index()
    if dotmode == '3d':
        df = df[df['dur'] > df['dur'].min()]
        df = df[df['dur'] > df['dur'].min()]

    ptA = df1[['x0','x1','b0','b1','m0','m1']].median()
    df = df[df['dur'] <= np.exp(ptA['x1'])] # ignore higher durs since line there is flat
    df['y1'] = df['dur']**ptA['m0']*np.exp(ptA['b0'])
    x1 = df['dur'] > np.exp(ptA['x0'])
    df['y1'][x1] = df['dur'][x1]**ptA['m1']*np.exp(ptA['b1'])

    ptB = df2[['b0','m0']].median()
    df['y2'] = df['dur']**ptB['m0']*np.exp(ptB['b0'])

    sz = 80
    lw1 = 1
    lw2 = 2
    df['y1e'] = -(df['y1'] - df['mean'])/df['std']
    df['y2e'] = -(df['y2'] - df['mean'])/df['std']
    # plt.plot(df['dur'], df['y2e'], lw=lw1, c='k', zorder=4)
    # plt.plot(df['dur'], df['y1e'], lw=lw1, c='k', zorder=4)
    plt.gca().fill_between(df['dur'], 0.0, df['y2e'], lw=0, color=colorB, facecolor=colorB, alpha=1.0, zorder=3)
    plt.gca().fill_between(df['dur'], 0.0, df['y1e'], lw=0, color=colorA, facecolor=colorA, alpha=0.6, zorder=3)
    plt.scatter(df['dur'], df['y2e'], sz, c=colorB, lw=lw1, label='bi-limb fit', zorder=5)
    plt.scatter(df['dur'], df['y1e'], sz, c=colorA, lw=lw1, label='tri-limb fit', zorder=5)

    plt.plot([np.exp(ptA['x0']), np.exp(ptA['x0'])], [-29.5, 8.5], '--', lw=lw2, c=colorA, zorder=1)
    plt.plot([0.01, df['dur'].max() + 50], [0, 0], '-', lw=lw2, c='k', zorder=1)

    # plt.plot(df['dur'], df['mean'])
    # plt.scatter(df['dur'], df['y1'], c='g')
    # plt.scatter(df['dur'], df['y2'], c='r')
    # plt.yscale('log')
    ys = np.hstack([df['y1e'].values, df['y2e'].values])
    plt.xscale('log')
    # plt.xlim(df0a['dur'].min()-2, df['dur'].max()+30)
    # plt.ylim(ys.min()-1, ys.max()+1)
    plt.xlabel('Duration (msec)')
    plt.ylabel('Motion sensitivity residual')
    leg = plt.legend(loc='upper left', prop={'size': 14})
    leg.get_frame().set_linewidth(1.5)
    xtcks = np.array([33, 200, 1000])
    # xtcks = xtcks[xtcks >= df['dur'].min()]
    plt.xticks(xtcks, xtcks)

    # formatting
    plt.minorticks_off()
    plt.gca().tick_params(top='off')
    plt.gca().tick_params(right='off')
    plt.gca().tick_params(axis='x', direction='out', width=1.5, length=5)
    plt.gca().tick_params(axis='y', direction='out', width=1.5, length=5)
    plt.gcf().patch.set_facecolor('white')
    for axis in ['bottom', 'left']:
        plt.gca().spines[axis].set_linewidth(lw2)
    for axis in ['top', 'right']:
        plt.gca().spines[axis].set_linewidth(0)

def main():
    yrng = (-17.0, 10.0)
    xrng = (30.0, 1025.0)
    df0, df1, df2 = load()
    plot(df0, df1, df2, '2d', colorA=[0.65, 0.65, 0.9], colorB=[0.2, 0.4, 1.0])
    plt.xlim(xrng)
    plt.ylim(yrng)
    # plt.show()
    plt.savefig('../plots/elbowResiduals-ALL-2d.pdf')
    plt.clf()
    plot(df0, df1, df2, '3d', colorA=[0.9, 0.65, 0.65], colorB=[0.9, 0.1, 0.1])
    plt.xlim(xrng)
    plt.ylim(yrng)
    # plt.show()
    plt.savefig('../plots/elbowResiduals-ALL-3d.pdf')

if __name__ == '__main__':
    main()
