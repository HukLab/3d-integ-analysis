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

def solve_all_durations(df, coh_thresh=0.75):
    threshes, thetas = {}, {}
    for di in df['duration_index'].unique():
        dfc = df[df['duration_index'] == di]
        xs = dfc['coherence'].values
        ys = dfc['correct'].values.astype('float')
        thetas[di] = solve(xs, ys)
        threshes[di] = inv_weibull(thetas[di], coh_thresh) if thetas[di] is not None else None
        if thetas[di] is None:
            print 'ERROR   (di={0})'.format(di)
        else:
            print 'SUCCESS (di={0}): {1}'.format(di, thetas[di])
    return threshes, thetas

def main_fit(args):
    df = default_filter_df(load(args))
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        threshes, thetas = solve_all_durations(df_dotmode)
        xs, ys = zip(*[(durmap[di], threshes[di]) for di in sorted(threshes) if threshes[di] is not None])
        slope, intercept, r_value, p_value, std_err = linregress(np.log(xs), np.log(ys))
        print '{4}: slope={0}, intercept={1}, r^2={2}, s.e.={3}'.format(slope, intercept, r_value**2, std_err, dotmode)
        plt.plot(np.log(xs), slope*np.log(xs) + intercept, color='g' if dotmode == '2d' else 'r')
        plt.scatter(np.log(xs), np.log(ys), label=dotmode, color='g' if dotmode == '2d' else 'r')
    plt.title('75% coh thresh vs. duration')
    plt.xlabel('log(duration)')
    plt.ylabel('log(75% coh thresh)')
    plt.legend()
    plt.show()

def plot_raw(df, dotmode):
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = df['coherence'].unique()
    colmap = dict((coh, col) for coh, col in zip(cohs, color_list(len(cohs))))

    fig = plt.figure()
    ax = plt.subplot(111)
    for coh, df_coh in df.groupby('coherence'):
        isp, ysp = zip(*df_coh.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [durmap[i] for i in isp]
        ax.plot(xsp, ysp, color=colmap[coh], label="%0.2f" % coh, marker='o', linestyle='-')
    plt.title('{0}: % correct vs. duration, by coherence'.format(dotmode))
    plt.xlabel('duration')
    plt.ylabel('% correct')
    plt.xscale('log')
    plt.xlim([0.035, 1.05])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def main_raw(args):
    df = load(args)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        plot_raw(df_dotmode, dotmode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    parser.add_argument('--fit', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode}
    if args.fit:
        main_fit(ps)
    else:
        main_raw(ps)
