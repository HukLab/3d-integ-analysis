import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from sample import bootstrap
from settings import nsigdots, COLOR_MAP
from weibull import weibull, inv_weibull, solve

def bootstrap_solve(xs0, ys0, unfold, nboots, thresh_val):
    guess = (1500, 7) if unfold else (85, 0.75)
    zss = bootstrap(zip(xs0, ys0), nboots)
    thetas, threshes = [], []
    for i in xrange(nboots+1):
        if i == 0:
            xs = np.array(xs0)
            ys = np.array(ys0)
        else:
            xs = zss[i-1][:, 0]
            ys = zss[i-1][:, 1]
        theta = theta = solve(xs, ys, unfold, guess=guess, quick=True)
        thresh = inv_weibull(theta, thresh_val, unfold) if theta is not None else None
        thetas.append(theta)
        threshes.append(thresh)
    return thetas, threshes

def main(args, nboots, unfold, savefig, outdir, thresh_val=0.75):
    df = load(args)
    subjs = df['subj'].unique()
    w = (2*(df.direction>2)-1) if unfold else 1
    df['nsigdots'] = w*nsigdots(df.coherence, df.duration)
    if unfold:
        min_nsigdots = df['nsigdots'].min()
        df['nsigdots'] = df['nsigdots'] - min_nsigdots
    df['nsigdots_binned'] = (df['nsigdots']/15).astype('int')*15
    df['detected'] = (df['response']==1) if unfold else df['correct']
    for dotmode, dfc in df.groupby('dotmode'):
        xs, ys = zip(*dfc[['nsigdots', 'detected']].sort('nsigdots').values)
        ys = np.array(ys).astype('float')
        thetas, threshes = bootstrap_solve(xs, ys, unfold, nboots, thresh_val)
        for theta, thresh in zip(thetas, threshes):
            if theta is not None:
                print theta, thresh
                plt.plot(xs, weibull(xs, theta, unfold), color=COLOR_MAP[dotmode], linestyle='-')
                if len(threshes) == 1:
                    pass
                    # plt.axvline(thresh, color=COLOR_MAP[dotmode], linestyle='--')
                    # plt.text(thresh + 1, 0.5 if dotmode == '2d' else 0.47, 'threshold={0} dots'.format(int(thresh)), color=COLOR_MAP[dotmode])
        xsb, ysb = zip(*dfc.groupby('nsigdots_binned', as_index=False)['detected'].agg(np.mean).sort('nsigdots_binned').values)
        plt.scatter(xsb, ysb, s=4, label=dotmode, color=COLOR_MAP[dotmode])
    plt.xlabel('nsigdots')
    plt.ylabel('% {0}'.format('right response' if unfold else 'correct'))
    if unfold:
        ax = plt.gca()
        rnd_10 = lambda x: int(10 * round(float(x)/10))
        min_xtick = rnd_10(df['nsigdots'].min() + min_nsigdots)
        max_xtick = rnd_10(df['nsigdots'].max() + min_nsigdots)
        xticks = np.linspace(min_xtick - min_nsigdots, max_xtick - min_nsigdots, 10)
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels([int(x + min_nsigdots) for x in xticks])
    else:
        # plt.xscale('log')
        plt.ylim([0.45, 1.05])
    plt.xlim([None, None])
    subj_label = subjs[0] if len(subjs) == 0 else 'ALL'
    plt.title(subj_label)
    if savefig:
        plt.savefig(os.path.join(outdir, '{0}-{1}.png'.format(subj_label, 'unfolded' if unfold else 'folded')))
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--nboots', required=False, type=int, default=0)
    parser.add_argument('--unfold', required=False, action='store_true', default=False)
    parser.add_argument('--savefig', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()
    main({'subj': args.subj, 'dotmode': args.dotmode}, args.nboots, args.unfold, args.savefig, args.outdir)
