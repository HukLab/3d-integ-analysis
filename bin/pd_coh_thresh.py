import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize

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

def solve(xs, ys, guess=(0.3, 1.0), ntries=5):
    neg_log_likelihood_fcn = lambda theta: weibull_mle(theta, xs, ys)
    for _ in xrange(ntries):
        soln = minimize(neg_log_likelihood_fcn, guess, method='TNC', bounds=[(0, None), (0, None)], constraints=[])
        if soln['success']:
            print 'SUCCESS (di={0}, dotmode={2}): {1}'.format(di, soln['x'], dotmode)
            theta_hat = soln['x']
            return theta_hat
            # plt.scatter(dfc.coherence.unique(), weibull(dfc.coherence.unique(), theta_hat))
            # plt.show()
    print 'ERROR   (di={0}, dotmode={2}): {1}'.format(di, soln['message'], dotmode)
    return None

def solve_all_durations(df, coh_thresh=0.75):
    threshes, thetas = {}, {}
    for di in df.duration_index.unique():
        dfc = df[df.duration_index == di]
        xs = dfc.coherence.values
        ys = dfc.correct.values.astype('float')
        thetas[di] = solve(xs, ys)
        threshes[di] = inv_weibull(thetas[di], coh_thresh)
    return threshes, thetas

def main():
    df = default_filter_df(load())
    durmap = dict(df.groupby('duration_index').duration.agg(min).reset_index().values)
    for dotmode in df.dotmode.unique():
        threshes, thetas = solve_all_durations(df[df.dotmode == dotmode])
        xs, ys = zip(*[(durmap[di], threshes[di]) for di in sorted(threshes)])
        slope, intercept, r_value, p_value, std_err = linregress(np.log(xs), np.log(ys))
        print '{4}: slope={0}, intercept={1}, r^2={2}, s.e.={3}'.format(slope, intercept, r_value**2, std_err, dotmode)
        plt.plot(np.log(xs), slope*np.log(xs) + intercept, color='g' if dotmode == '2d' else 'r')
        plt.scatter(np.log(xs), np.log(ys), label=dotmode, color='g' if dotmode == '2d' else 'r')
    plt.title('75% coh thresh vs. duration')
    plt.xlabel('log(duration)')
    plt.ylabel('log(75% coh thresh)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
