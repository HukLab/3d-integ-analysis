import operator
import itertools

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize

from tools import color_list
from pd_io import load, default_filter_df

def params(theta):
    if len(theta) == 2:
        a, b = theta
        maxV = 1.0
    elif len(theta) == 3:
        a, b, maxV = theta
    else:
        raise Exception("params must be length 2 or 3: {0}".format(theta))
    minV = 0.5
    return a, b, minV, maxV

def weibull(x, theta):
    """
    theta = scale, shape, maxV [optional, in case performance is less than 1]
    """
    a, b, minV, maxV = params(theta)
    return maxV - (maxV-minV) * np.exp(-pow(x/a, b))

def inv_weibull(theta, y):
    """
    the function calculates the inverse of a weibull function
    with given parameters (theta) for a given y value
    returns the x value
    """
    a, b, minV, maxV = params(theta)
    return a * pow(np.log((maxV-minV)/(maxV-y)), 1.0/b)

def weibull_mle(theta, xs, ys):
    yh = weibull(xs, theta)
    logL = np.sum(ys*np.log(yh) + (1-ys)*np.log(1-yh))
    if np.isnan(logL):
        yh = yh*0.99 + 0.005
        logL = np.sum(ys*np.log(yh) + (1-ys)*np.log(1-yh))
    return -logL

def solve(xs, ys, guess=(0.3, 1.0), ntries=10):
    neg_log_likelihood_fcn = lambda theta: weibull_mle(theta, xs, ys)
    for i in xrange(ntries):
        if i > 0:
            guess = (guess[0] + i/10.0, guess[1] + i/10.0)
        soln = minimize(neg_log_likelihood_fcn, guess, method='TNC', bounds=[(0, None), (0, None)], constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            return theta_hat
    return None

def solve_all_durations(df, thresh_val):
    threshes, thetas = {}, {}
    for di in df['duration_index'].unique():
        dfc = df[df['duration_index'] == di]
        xs = dfc['coherence'].values
        ys = dfc['correct'].values.astype('float')
        thetas[di] = solve(xs, ys)
        threshes[di] = inv_weibull(thetas[di], thresh_val) if thetas[di] is not None else None
        if thetas[di] is None:
            print 'ERROR   (di={0})'.format(di)
        else:
            print 'SUCCESS (di={0}): {1}, {2}'.format(di, thetas[di], np.log(threshes[di]))
    return threshes, thetas

def set_xticks(pts, ax):
    vals = np.array(list(itertools.chain(*[x for x,y in pts.values()])))
    xticks = np.log(1000*vals)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels([int(np.exp(x)) for x in xticks])

def set_yticks(pts, ax):
    vals = np.array(list(itertools.chain(*[y for x,y in pts.values()])))
    ys = np.log(vals)
    yticks = np.linspace(min(ys), max(ys), 10)
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels([int(100*np.exp(y)) for y in yticks])

def find_elbow(xs, ys, presets=None, ntries=10):
    xs = np.log(1000*np.array(xs))
    ys = np.log(ys)
    def error_fcn((x0, A0, B0, A1, B1)):
        z = np.array([xs < x0, xs*A0 + B0, xs*A1 + B1])
        yh = z[0]*z[1] + (1-z[0])*z[2]
        return np.sum(np.power(ys-yh, 2))
    bounds = [(min(xs), max(xs)), (None, 0), (None, None), (None, 0), (None, None)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.array([x[0]*(x[1] - x[3]) + x[2] - x[4]]) }]
    if presets is not None:
        assert len(presets) == 5
        for i, val in enumerate(presets):
            if val is not None:
                constraints.append({'type': 'eq', 'fun': lambda x, i=i, val=val: x[i] - val })
    guess = np.array([np.mean(xs), -1, 0, -0.5, 0])
    for i in xrange(ntries):
        soln = minimize(error_fcn, guess*(1 + i/10.), method='SLSQP', bounds=bounds, constraints=constraints)
        if soln['success']:
            theta_hat = soln['x']
            return theta_hat
    return None

def plot_and_fit_thresholds(pts, thresh_val, x_split_default=82, solve_elbow=True):
    # presets = (None, -1.0, None, -0.5, None)
    presets = None
    fig = plt.figure()
    ax = plt.subplot(111)
    for dotmode, (xs, ys) in pts.iteritems():
        color='g' if dotmode == '2d' else 'r'
        theta_hat = find_elbow(xs, ys, presets) if solve_elbow else None
        xs = 1000*np.array(xs)
        xs2, ys2 = [], []
        if theta_hat is not None:
            x_split = np.exp(theta_hat[0])
        else:
            x_split = x_split_default
        if x_split is not None:
            print 'Splitting xs at {0} ms'.format(x_split)
            assert all(operator.le(xs[i], xs[i+1]) for i in xrange(len(xs)-1)) # xs is sorted
            try:
                ind = next(i for i, x in enumerate(xs) if x > x_split) # index of first bin not including x_split
                xs2, ys2 = xs[ind-1:], ys[ind-1:] # actually want to include point before so lines meet
                xs, ys = xs[:ind], ys[:ind]
            except StopIteration: # all of xs is less than x_split
                pass
            plt.axvline(np.log(x_split), color=color, linestyle='--')
        for i, (x, y) in enumerate([(xs, ys), (xs2, ys2)]):
            if len(x) == 0:
                continue
            if theta_hat is not None:
                slope, intercept = theta_hat[2*i + 1], theta_hat[2*i + 2]
                print '{2}, {3}: slope={0}, intercept={1}'.format(slope, intercept, dotmode, i+1)
            else:
                slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
                print '{3}, {4}: slope={0}, intercept={1}, r^2={2}'.format(slope, intercept, r_value**2, dotmode, i+1)
            x0 = [min(x), x_split] if i == 0 else [x_split, max(x)]
            plt.plot(np.log(x0), slope*np.log(x0) + intercept, color=color)
            plt.scatter(np.log(x), np.log(y), label=dotmode, color=color)
            plt.text(np.mean(np.log(x0)), np.mean(np.log(y)), 'slope={0:.2f}'.format(slope), color=color)
    plt.title('{0}% coh thresh vs. duration'.format(int(thresh_val * 100)))
    set_xticks(pts, ax)
    set_yticks(pts, ax)
    plt.xlabel('duration (ms)')
    plt.ylabel('{0}% coh thresh'.format(int(thresh_val * 100)))
    plt.legend()
    plt.show()

def thresholds(args, thresh_val):
    df = default_filter_df(load(args))
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    pts = {}
    for dotmode, df_dotmode in df.groupby('dotmode'):
        print dotmode
        threshes, thetas = solve_all_durations(df_dotmode, thresh_val)
        xs, ys = zip(*[(durmap[di], threshes[di]) for di in sorted(threshes) if threshes[di] is not None])
        pts[dotmode] = (xs, ys)
    plot_and_fit_thresholds(pts, thresh_val)

def plot(df, dotmode):
    durinds = df['duration_index'].unique()
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cols = color_list(len(durinds))
    colmap = dict((di, col) for di, col in zip(durinds, cols))

    fig = plt.figure()
    ax = plt.subplot(111)
    for di in durinds:
        dfc = df[df['duration_index'] == di]
        xsp, ysp = zip(*dfc.groupby('coherence').agg(np.mean)['correct'].reset_index().values)
        ax.plot(xsp, ysp, color=colmap[di], label="%0.2f" % durmap[di], marker='o', linestyle='-')
    plt.title('{0}: % correct vs. coherence, by duration'.format(dotmode))
    plt.xlabel('coherence')
    plt.ylabel('% correct')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def main(args):
    df = load(args)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        plot(df_dotmode, dotmode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    parser.add_argument('--thresh', action='store_true', default=False)
    parser.add_argument('--thresh-val', type=float, default=0.75)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    if args.thresh:
        thresholds(ps, args.thresh_val)
    else:
        main(ps)
