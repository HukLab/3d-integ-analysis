import operator
import itertools

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize

from sample import bootstrap
from tools import color_list
from pd_io import load, default_filter_df

def make_durmap(df):
    return dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)

def make_colmap(durinds):
    cols = color_list(len(durinds))
    return dict((di, col) for di, col in zip(durinds, cols))

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

def solve(xs, ys, guess=(0.3, 1.0), ntries=20):
    neg_log_likelihood_fcn = lambda theta: weibull_mle(theta, xs, ys)
    for i in xrange(ntries):
        if i > 0:
            guess = (guess[0] + i/10.0, guess[1] + i/10.0)
        soln = minimize(neg_log_likelihood_fcn, guess, method='TNC', bounds=[(0, None), (0, None)], constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            return theta_hat
    return None

def plot_weibull_with_data(dfc, dotmode, theta, thresh, color, dur, ax, show_for_each_dotmode, is_boot):
    xsp, ysp = zip(*dfc.groupby('coherence', as_index=False)['correct'].agg(np.mean).values)
    if show_for_each_dotmode:
        label = "%0.2f" % dur
    else:
        color = 'g' if dotmode == '2d' else 'r'
        label = dotmode
    if is_boot:
        label = ''
    ax.scatter(xsp, ysp, color=color, label=label, marker='o')
    xsc = np.linspace(min(xsp), max(xsp))
    if not show_for_each_dotmode:
        plt.axvline(thresh, color=color, linestyle='--')
        plt.text(thresh + 0.01, 0.5 if dotmode == '2d' else 0.45, 'threshold={0}%'.format(int(thresh*100)), color=color)
    ax.plot(xsc, weibull(xsc, theta), color=color, linestyle='-')

def solve_one_duration(xs, ys, di, thresh_val):
    theta = solve(xs, ys)
    thresh = inv_weibull(theta, thresh_val) if theta is not None else None
    if theta is None:
        pass # print 'ERROR   (di={0})'.format(di)
    else:
        pass # print 'SUCCESS (di={0}): {1}, {2}'.format(di, theta, np.log(thresh))
    return theta, thresh

def solve_all_durations(df, dotmode, nboots, thresh_val, ax, show_for_each_dotmode):
    durinds = sorted(df['duration_index'].unique())
    colmap = make_colmap(durinds)
    durmap = make_durmap(df)
    threshes, thetas = dict(zip(durinds, [list() for _ in xrange(len(durinds))])), dict(zip(durinds, [list() for _ in xrange(len(durinds))]))
    for di, df_durind in df.groupby('duration_index'):
        zss = [bootstrap(y.values, nboots) for x,y in df_durind[['coherence','correct']].sort('coherence').groupby('coherence')]
        zss = [np.vstack([z[i] for z in zss]) for i in xrange(nboots)]
        print di
        for i in xrange(nboots+1):
            if i == 0:
                xs, ys = zip(*df_durind[['coherence','correct']].values)
                xs = np.array(xs)
                ys = np.array(ys).astype('float')
            else:
                xs = zss[i-1][:, 0].astype('float')
                ys = zss[i-1][:, 1].astype('float')
            theta, thresh = solve_one_duration(xs, ys, di, thresh_val)
            if theta is not None and ax is not None:
                plot_weibull_with_data(df_durind, dotmode, theta, thresh, colmap[di], durmap[di], ax, show_for_each_dotmode, i>0)
            thetas[di].append(theta)
            threshes[di].append(thresh)
    return threshes, thetas

def solve_ignoring_durations(df, dotmode, nboots, thresh_val, ax):
    di = -1
    threshes, thetas = [], []
    zss = [bootstrap(y.values, nboots) for x,y in df[['coherence','correct']].sort('coherence').groupby('coherence')]
    zss = [np.vstack([z[i] for z in zss]) for i in xrange(nboots)]
    for i in xrange(nboots+1):
        if i == 0:
            xs, ys = zip(*df[['coherence','correct']].values)
            xs = np.array(xs)
            ys = np.array(ys).astype('float')
        else:
            xs = zss[i-1][:, 0].astype('float')
            ys = zss[i-1][:, 1].astype('float')
        theta, thresh = solve_one_duration(xs, ys, di, thresh_val)
        if theta is not None and ax is not None:
            plot_weibull_with_data(df, dotmode, theta, thresh, 'g' if dotmode == '2d' else 'r', dotmode, ax, False, i>0)
        thetas.append(theta)
        threshes.append(thresh)
    return threshes, thetas

is_nan_or_inf = lambda items: np.isnan(items) | np.isinf(items)
def remove_nan_or_inf(items):
    items = np.array(items)
    return items[~is_nan_or_inf(items)]

def set_xticks(pts, ax):
    vals = np.array(list(itertools.chain(*[x for x,y in pts.values()])))
    xticks = np.log(1000*vals)
    xticks = list(set(remove_nan_or_inf(xticks)))
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels([int(np.exp(x)) for x in xticks])

def set_yticks(pts, ax):
    vals = np.array(list(itertools.chain(*[y for x,y in pts.values()])))
    ys = np.log(vals)
    ys = remove_nan_or_inf(ys)
    yticks = np.linspace(min(ys), max(ys), 10)
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels([int(100*np.exp(y)) for y in yticks])

def find_elbow(xs, ys, presets=None, ntries=10):
    xs = np.log(1000*np.array(xs))
    ys = np.log(ys)
    zs = np.array([xs, ys])
    xs, ys = zs[:, ~is_nan_or_inf(zs[0]) & ~is_nan_or_inf(zs[1])]
    def error_fcn((x0, A0, B0, A1, B1)):
        z = np.array([xs < x0, xs*A0 + B0, xs*A1 + B1])
        yh = z[0]*z[1] + (1-z[0])*z[2]
        return np.sum(np.power(ys-yh, 2))
    x0_max_ms = 0
    bounds = [(min(xs), np.log(np.exp(max(xs))-x0_max_ms)), (None, 0), (None, None), (None, 0), (None, None)]
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
        elif not solve_elbow:
            x_split = x_split_default
        else:
            x_split = None
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
        else:
            xs2 = []
            ys2 = []
        for i, (x, y) in enumerate([(xs, ys), (xs2, ys2)]):
            if len(x) == 0:
                continue
            if theta_hat is not None:
                slope, intercept = theta_hat[2*i + 1], theta_hat[2*i + 2]
                print '{2}, {3}: slope={0}, intercept={1}'.format(slope, intercept, dotmode, i+1)
            else:
                slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
                print '{3}, {4}: slope={0}, intercept={1}, r^2={2}'.format(slope, intercept, r_value**2, dotmode, i+1)
            if x_split is not None:
                ws = pd.DataFrame({'dur': x, 'pcor': y}).groupby('dur').agg([np.mean, np.std])
                ws['pcor_lower'] = ws['pcor']['mean'] - ws['pcor']['std']
                ws['pcor_upper'] = ws['pcor']['mean'] + ws['pcor']['std']
                x, y, yerr = ws.index.values.astype('float'), ws['pcor']['mean'].values, ws['pcor']['std'].values
                yerr = zip(*np.log(ws[['pcor_lower', 'pcor_upper']]).values)
                x0 = [min(x), x_split] if i == 0 else [x_split, max(x)]
                plt.plot(np.log(x0), slope*np.log(x0) + intercept, color=color)
                plt.errorbar(np.log(x), np.log(y), yerr=yerr, marker='o', linestyle='', label=dotmode if i==0 else '', color=color)
                plt.text(np.mean(np.log(x0)), np.mean(np.log(y)), 'slope={0:.2f}'.format(slope), color=color)
    plt.title('{0}% coh thresh vs. duration'.format(int(thresh_val * 100)))
    set_xticks(pts, ax)
    set_yticks(pts, ax)
    plt.xlabel('duration (ms)')
    plt.ylabel('{0}% coh thresh'.format(int(thresh_val * 100)))
    plt.legend()
    plt.show()

def plot_info(ax, label):
    plt.title('{0}: % correct vs. coherence'.format(label))
    plt.xlabel('coherence')
    plt.ylabel('% correct')
    # plt.xlim([0.0, 1.05])
    plt.xscale('log')
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def thresholds(args, nboots, plot_thresh, show_for_each_dotmode, thresh_val):
    df = default_filter_df(load(args))
    ndurinds = len(df['duration_index'].unique())
    subjs = df['subj'].unique()
    if ndurinds == 1:
        show_for_each_dotmode = False
    if not show_for_each_dotmode:
        fig = plt.figure()
        ax = plt.subplot(111)
    durmap = make_durmap(df)
    durmap[-1] = 'all'
    pts = {}
    for dotmode, df_dotmode in df.groupby('dotmode'):
        print dotmode
        if show_for_each_dotmode:
            fig = plt.figure()
            ax = plt.subplot(111)
        elif ndurinds > 1:
            _, _ = solve_ignoring_durations(df_dotmode, dotmode, nboots, thresh_val, ax)
            continue
        threshes, thetas = solve_all_durations(df_dotmode, dotmode, nboots, thresh_val, ax, show_for_each_dotmode)
        xs, ys = zip(*[(durmap[di], thresh) for di in sorted(threshes) for thresh in threshes[di]])
        pts[dotmode] = (xs, ys)
        if show_for_each_dotmode:
            plot_info(ax, '{0}'.format(dotmode) + (', {0}'.format(subjs[0].upper()) if len(subjs) == 1 else ''))
    if not show_for_each_dotmode and ndurinds == 1:
        plot_info(ax, "duration=%0.2fs" % durmap[df['duration_index'].unique()[0]] + (', {0}'.format(subjs[0].upper()) if len(subjs) == 1 else ''))
    elif not show_for_each_dotmode:
        plot_info(ax, "all durations" + (', {0}'.format(subjs[0].upper()) if len(subjs) == 1 else ''))
    if plot_thresh:
        plot_and_fit_thresholds(pts, thresh_val)

def plot(df, dotmode):
    durinds = sorted(df['duration_index'].unique())
    colmap = make_colmap(durinds)
    durmap = make_durmap(df)

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
    parser.add_argument('--durind', required=False, type=int)
    parser.add_argument('--thresh', action='store_true', default=False)
    parser.add_argument('--nboots', required=False, type=int, default=0)
    parser.add_argument('--plot-thresh', action='store_true', default=False)
    parser.add_argument('--join-dotmode', action='store_true', default=False)
    parser.add_argument('--thresh-val', type=float, default=0.75)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode, 'duration_index': args.durind}
    if args.thresh:
        thresholds(ps, args.nboots, args.plot_thresh, not args.join_dotmode, args.thresh_val)
    else:
        main(ps)
