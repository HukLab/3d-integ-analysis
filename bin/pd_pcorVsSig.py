import argparse
import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from sample import bootstrap
from session_info import nsigdots, COLOR_MAP
from weibull import weibull, inv_weibull, solve

def bootstrap_solve(xs0, ys0, unfold, nboots, thresh_val):
    zss = bootstrap(zip(xs0, ys0), nboots)
    thetas, threshes = [], []
    for i in xrange(nboots+1):
        if i == 0:
            xs = np.array(xs0)
            ys = np.array(ys0)
        else:
            xs = zss[i-1][:, 0]
            ys = zss[i-1][:, 1]
        theta = theta = solve(xs, ys, unfold, quick=False)
        thresh = inv_weibull(theta, thresh_val, unfold) if theta is not None else None
        thetas.append(theta)
        threshes.append(thresh)
    return thetas, threshes

def main(args, nboots, unfold, thresh_val=0.75):
    df = load(args)
    subjs = df['subj'].unique()
    w = (2*(df.direction>2)-1) if unfold else 1
    df['nsigdots'] = w*nsigdots(df.coherence, df.duration)
    df['nsigdots_binned'] = (df['nsigdots']/15).astype('int')*15
    df['detected'] = (df['response']==1) if unfold else df['correct']
    for dotmode, dfc in df.groupby('dotmode'):
        xs, ys = zip(*dfc[['nsigdots', 'detected']].sort('nsigdots').values)
        # xs = np.array(xs) - min(xs)
        # xs = (np.array(xs) - min(xs)) / (max(xs) - min(xs))
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
        # xsb = np.array(xsb) - min(xsb)
        # xsb = (np.array(xsb) - min(xsb)) / (max(xsb) - min(xsb))
        plt.scatter(xsb, ysb, s=4, label=dotmode, color=COLOR_MAP[dotmode])
    plt.xlabel('nsigdots')
    plt.ylabel('% {0}'.format('right response' if unfold else 'correct'))
    if not unfold:
        plt.xscale('log')
        plt.ylim([0.45, 1.05])
    plt.xlim([None, None])
    plt.title(subjs[0] if len(subjs) == 0 else 'ALL')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subj", required=False, type=str, help="")
    parser.add_argument("-d", "--dotmode", required=False, type=str, help="")
    parser.add_argument('--nboots', required=False, type=int, default=0)
    parser.add_argument('--unfold', required=False, action='store_true', default=False)
    args = parser.parse_args()
    main({'subj': args.subj, 'dotmode': args.dotmode}, args.nboots, args.unfold)
