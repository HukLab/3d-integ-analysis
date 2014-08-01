import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pd_io import load
from weibull import weibull
from tools import color_list

delay = 0.03
stim_power = lambda r, p: r[0]*((r[1]-delay)**p)

def log_likelihood(data, fcn, inner_fcn, thetas, fcn_kwargs):
    """
    data is array of [(x0, y0), (x1, y1), ...], where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi

    calculates the sum of the log-likelihood of data
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    likelihood = lambda row: fcn(inner_fcn(row[0], thetas[0]), thetas[1:], *fcn_kwargs) if row[1] else 1-fcn(inner_fcn(row[0], thetas[0]), thetas[1:], *fcn_kwargs)
    log_likeli = lambda row: np.log(likelihood(row))
    return sum(map(log_likeli, data))

# def solve(xs, ys, unfold=False, guess=(0.5, 0.5, 0.7), ntries=20, quick=True):
def solve(xs, ys, unfold=False, guess=(0.5, 0.5, 0.7, 0.5, 0.85), ntries=20, quick=True):
    guess = np.array(guess)
    APPROX_ZERO, APPROX_ONE = 0.00001, 0.99999
    bounds = [(APPROX_ZERO, None), (APPROX_ZERO, None), (APPROX_ZERO, None)] + [(APPROX_ZERO, APPROX_ONE)] * (len(guess) - 3)
    pf = lambda th, d=zip(xs, ys): -log_likelihood(d, weibull, stim_power, th, {'unfold': unfold})

    sol = None
    ymin = 100000
    method = 'L-BFGS-B' # 'SLSQP' 
    for i in xrange(ntries):
        if i > 0:
            guess = guess*np.random.uniform(0.95, 1.05)
        soln = minimize(pf, guess, method=method, bounds=bounds, constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            if not quick and soln['fun'] < ymin:
                sol = theta_hat
            else:
                return theta_hat
        else:
            print soln
    return sol

res = {'2d': [0.73003923, 0.04781754,  0.99229569,  0.46343525,  0.99357894], '3d': [ 0.71499905,  0.12562435,  1.25349538,  0.51364108,  0.95234274]} # no delay
res2 = {'2d': [ 0.72972991,  0.01550323,  0.48585816], '3d': [ 0.79408596,  0.02783787,  0.30990086]} # no delay
res3 = {'2d': [ 0.60015036,  0.04877081,  0.9787139,   0.45510507,  0.99413209], '3d': [ 0.6115084,   0.12791501,  1.19893156,  0.5092267,   0.95576519]} # delay=0.03
def plot(args=None, rs=res):
    df = load(args)
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = sorted(df['coherence'].unique())
    colmap = dict((coh, col) for coh, col in zip([0]*2 + cohs, color_list(len(cohs) + 2, "YlGnBu")))
    for dotmode, dfc0 in df.groupby('dotmode'):
        plt.figure()
        th = np.array(rs[dotmode])
        for coh, dfc in dfc0.groupby('coherence'):

            ib, yb = zip(*dfc.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
            xb = np.array([durmap[i] for i in ib])

            fi = coh*((xb-delay)**th[0])
            yf = weibull(fi, th[1:])

            color = colmap[coh] #'r' if dotmode == '3d' else 'g'
            label = '{0}'.format(coh)
            plt.plot(xb, yb, color=color, label=label, marker='o', linestyle='')
            plt.plot(xb, yf, color=color, linestyle='-')
        plt.xlabel('duration')
        plt.ylabel('% correct')
        plt.title(dotmode)
        plt.legend(loc='lower right')
        plt.savefig('../' + dotmode + '-vsDur.png')
        # plt.show()
    durs = sorted(df['duration_index'].unique())
    colmap = dict((dur, col) for dur, col in zip([0]*2 + durs, color_list(len(durs) + 2)))
    for dotmode, dfc0 in df.groupby('dotmode'):
        plt.figure()
        th = np.array(rs[dotmode])
        for di, dfc in dfc0.groupby('duration_index'):
            dur = durmap[di]
            xb, yb = zip(*dfc.groupby('coherence').agg(np.mean)['correct'].reset_index().values)
            # xb = np.array([durmap[i] for i in ib])

            fi = np.array(xb)*((dur-delay)**th[0])
            yf = weibull(fi, th[1:])

            color = colmap[di] #'r' if dotmode == '3d' else 'g'
            label = '{0:0.2f}'.format(dur)
            plt.plot(xb, yb, color=color, label=label, marker='o', linestyle='')
            plt.plot(xb, yf, color=color, linestyle='-')
        plt.xscale('log')
        plt.xlabel('coherence')
        plt.ylabel('% correct')
        plt.legend(loc='upper left')
        plt.title(dotmode)
        plt.savefig('../' + dotmode + '-vsCoh.png')
        # plt.show()
    # plt.show()

def main(args=None):
    df = load(args)
    for dotmode, dfc in df.groupby('dotmode'):
        pts = dfc[['coherence', 'duration', 'correct']].values
        xs, ys = zip(*np.array([((x,y), z) for x,y,z in pts]))
        th = solve(xs, ys)
        print th

if __name__ == '__main__':
    plot()
    # main()
