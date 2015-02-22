import os.path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from mle import APPROX_ZERO, mle
from saturating_exponential import saturating_exp, double_saturating_exp

is_nan_or_inf = lambda items: np.isnan(items) | np.isinf(items)

def load_all_sens(indir, subjs):
    f1 = lambda subj: 'pcorVsCohByDur_thresh-{0}-params.csv'.format(subj)
    fn = lambda indir, fn: os.path.abspath(os.path.join(indir, fn))
    df = pd.concat([pd.read_csv(fn(indir, f1(subj))) for subj in subjs])

    # keep only non-zero sensitivities
    df = df[(df['sens'] >= 0.0)]# & (df['sens'] < 150.0)]

    # remove first two duration bins for 3d
    inds = (df['dotmode']=='3d') & (df['di'] < 3)
    df = df[~inds]
    # remove first two duration bins for 2d as well
    inds = (df['dotmode']=='2d') & (df['di'] < 3)
    df = df[~inds]
    return df

def xs_ys(xs, ys, f=None):
    if f is None:
        f = lambda x: x
    zs = np.array([f(xs), f(ys)])
    xs, ys = zs[:, ~is_nan_or_inf(zs[0]) & ~is_nan_or_inf(zs[1])]
    return xs, ys

def fit_subjs_single_sat_exp(indir, outdir, subjs):
    df = load_all_sens(indir, subjs)
    Af = {'2d': 25.707777, '3d': 8.625575}
    Bf = {'2d': 4.484605, '3d': 2.591813}
    # T1 = 100.
    df['dur'] = df['dur'] / 1000.0
    x0f = {'2d': 2./30, '3d': 2./30}
    # x0f = {'2d': 1./30, '3d': 2./30}
    T1 = 200.
    # bounds = [(APPROX_ZERO, 40.0), (APPROX_ZERO, None), (APPROX_ZERO, None)]
    # cons = [{'type': 'ineq', 'fun': lambda x: x[0] - x[1]}]
    # bounds = [(APPROX_ZERO, 40.0), (APPROX_ZERO, None)]
    bounds = [(APPROX_ZERO, None)]
    cons = []

    vals = []
    for (dotmode, subj), dfc in df.groupby(['dotmode', 'subj']):
        # A = dfc[dfc['di'] >= dfc['di'].max() - 2]['sens'].median() # avg of last 3 pts
        # B = dfc[dfc['di'] == dfc['di'].min()]['sens'].median()
        A, B = Af[dotmode], Bf[dotmode]
        x0 = x0f[dotmode]
        for bi, dfcur in dfc.groupby('bi'):
            xs, ys = xs_ys(dfcur['dur'].values, dfcur['sens'].values)
            yh = lambda (t1): saturating_exp(xs, A, B, t1, x0=x0)
            # yh = lambda (a, b, t1): saturating_exp(xs, a, b, t1, x0=x0)
            # yh = lambda (a, t1): saturating_exp(xs, a, B, t1, x0=x0)
            # yh = lambda (b, t1): saturating_exp(xs, A, b, t1, x0=x0)
            obj = lambda theta: ((ys - yh(theta))**2).sum()
            th = mle([1], obj, guesses=[(T1)], bounds=bounds, constraints=cons, quick=True, method='SLSQP', opts={'maxiter': 200})
            # th = mle([1], obj, guesses=[(A, T1)], bounds=bounds, constraints=cons, quick=True, method='SLSQP', opts={'maxiter': 200})
            # th = mle([1], obj, guesses=[(B, T1)], bounds=bounds, constraints=cons, quick=True, method='SLSQP', opts={'maxiter': 200})
            if th and th[-1]['success']:
                t1 = th[-1]['x'][0]
                sse = obj((t1))
                vals.append((subj, dotmode, bi, A, B, x0, t1, 0.0, 1.0, sse))

                # a, t1 = th[-1]['x']
                # sse = obj((a, t1))
                # vals.append((subj, dotmode, bi, a, B, x0, t1, 0.0, 1.0, sse))
                
                # a, b, t1 = th[-1]['x']
                # sse = obj((a, b, t1))
                # vals.append((subj, dotmode, bi, a, b, x0, t1, 0.0, 1.0, sse))
                
                # b, t1 = th[-1]['x']
                # sse = obj((b, t1))
                # vals.append((subj, dotmode, bi, A, b, x0, t1, 0.0, 1.0, sse))
    df0 = pd.DataFrame(vals, columns=['subj', 'dotmode', 'bi', 'A', 'B', 'x0', 'T1', 'T2', 'p', 'sse'])
    return df0

def fit_subjs_double_sat_exp(indir, outdir, subjs):
    df = load_all_sens(indir, subjs)
    df['dur'] = df['dur'] / 1000.0
    x0f = {'2d': 2./30, '3d': 2./30}

    # x0f = {'2d': 100./3, '3d': 200./3}
    Af = {'2d': 25.707777, '3d': 8.625575}
    Bf = {'2d': 4.484605, '3d': 2.591813}
    T1 = 100.
    T2 = 1000.
    p = 0.5
    bounds = [(50.0, 500.0), (500.0, None), (APPROX_ZERO, 1.0-APPROX_ZERO)]
    # bounds = [(APPROX_ZERO, None), (APPROX_ZERO, None), (APPROX_ZERO, 1.0-APPROX_ZERO)]
    cons = [{'type': 'ineq', 'fun': lambda x: x[1] - x[0]}]

    vals = []
    for (dotmode, subj), dfc in df.groupby(['dotmode', 'subj']):
        A, B = Af[dotmode], Bf[dotmode]
        # A = dfc[dfc['di'] == dfc['di'].max()]['sens'].median()
        # A = dfc[dfc['di'] >= dfc['di'].max() - 2]['sens'].median() # avg of last 3 pts
        # B = dfc[dfc['di'] == dfc['di'].min()]['sens'].median()
        x0 = x0f[dotmode]
        for bi, dfcur in dfc.groupby('bi'):
            xs, ys = xs_ys(dfcur['dur'].values, dfcur['sens'].values)
            yh = lambda (t1, t2, p): double_saturating_exp(xs, A, B, t1, t2, p, x0=x0)
            obj = lambda theta: ((ys - yh(theta))**2).sum()
            th = mle([1], obj, guesses=[(T1, T2, p)], bounds=bounds, constraints=cons, quick=True, method='SLSQP', opts={'maxiter': 200})
            if th and th[-1]['success']:
                t1, t2, P = th[-1]['x']
                sse = obj((t1, t2, P))
                vals.append((subj, dotmode, bi, A, B, x0, t1, t2, P, sse))
    df0 = pd.DataFrame(vals, columns=['subj', 'dotmode', 'bi', 'A', 'B', 'x0', 'T1', 'T2', 'p', 'sse'])
    return df0

def fit_line(dfc, lb, ub):
    msk = (dfc['dur'] >= lb) & (dfc['dur'] <= ub)
    xs, ys = xs_ys(dfc[msk]['dur'].values, dfc[msk]['sens'].values, f=np.log)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    return slope, intercept, r_value, p_value, std_err

def slope_quantiles(df, delta=0.34):
    df0 = df.groupby(['dotmode','subj']).quantile([0.5-delta, 0.5, 0.5+delta])['slope'].reset_index()
    vals = []
    for i, (subj, dfc) in enumerate(df0.groupby('subj')):
        lb2, med2, ub2 = dfc[dfc['dotmode']=='2d']['slope'].values
        lb3, med3, ub3 = dfc[dfc['dotmode']=='3d']['slope'].values
        vals.append(('2d', subj, lb2, med2, ub2, delta))
        vals.append(('3d', subj, lb3, med3, ub3, delta))
    df1 = pd.DataFrame(vals, columns=['dotmode', 'subj', 'lb', 'med', 'ub', 'delta'])
    return df1

def fit_subjs_decision_slope(indir, outdir, subjs):
    df = load_all_sens(indir, subjs)

    elb1 = {'2d': (86.0, 149.0), '3d': (141.0, 194.0)}
    elb2 = {'2d': (983.0, 1040.0), '3d': (1199.0, 3100.0)}
    elb1m = {'2d': 136.0, '3d': 171.0}
    elb2m = {'2d': 983.0, '3d': 1267.0}

    sdev_below = lambda mu, lb: mu-lb if mu > lb else 1.0
    sdev_above = lambda mu, ub: ub-mu if ub > mu else 1.0
    gauss_below = lambda mu, lb, sz=4: [stats.norm(mu, sdev_below(mu, lb)).ppf(i) for i in np.linspace(0.1, 0.4, sz)]
    gauss_above = lambda mu, ub, sz=5: [stats.norm(mu, sdev_above(mu, ub)).ppf(i) for i in np.linspace(0.5, 0.9, sz)]
    gauss = lambda mu, lb, ub: gauss_below(mu, lb) + gauss_above(mu, ub)
    lbs = dict((dotmode, gauss(elb1m[dotmode], *elb1[dotmode])) for dotmode in ['2d', '3d'])
    ubs = dict((dotmode, gauss(elb2m[dotmode], *elb2[dotmode])) for dotmode in ['2d', '3d'])
    
    vals = []
    for (dotmode, subj), dfc in df.groupby(['dotmode', 'subj']):
        if subj.upper() == 'ALL':
            continue
        print subj, dotmode
        lb0 = elb1m[dotmode]
        ub0 = elb2m[dotmode]
        for j, dfcur in dfc.groupby('bi'):
            slope, intercept, r_value, p_value, std_err = fit_line(dfcur, lb0, ub0)
            vals.append((subj, dotmode, 0, j, lb0, ub0, slope, intercept, r_value, p_value, std_err))
        i = 0
        for lb in lbs[dotmode]:
            for ub in ubs[dotmode]:
                for j, dfcur in dfc.groupby('bi'):
                    slope, intercept, r_value, p_value, std_err = fit_line(dfcur, lb, ub)
                    vals.append((subj, dotmode, i+1, j, lb, ub, slope, intercept, r_value, p_value, std_err))
                    i += 1
    df0 = pd.DataFrame(vals, columns=['subj', 'dotmode', 'ind', 'bi', 'lb', 'ub', 'slope', 'intercept', 'r_value', 'p_value', 'std_err'])
    return df0

def main(args):
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if args.task == 'fit-decision-slope':
        subjs = ['KRM', 'KLB', 'HUK', 'LKC', 'LNK']
        df = fit_subjs_decision_slope(args.indir, args.outdir, subjs)
    elif args.task == 'bin-decision-slope':
        df0 = pd.read_csv(os.path.join(args.indir, args.infile))
        df = slope_quantiles(df0)
    else:
        subjs = ['ALL']
        if args.task[-1] == '1':
            df = fit_subjs_single_sat_exp(args.indir, args.outdir, subjs)
        else:
            df = fit_subjs_double_sat_exp(args.indir, args.outdir, subjs)

    outfile = os.path.join(args.outdir, args.outfile)
    while os.path.exists(outfile):
        outfile = outfile.replace('.', '_.')
    df.to_csv(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str)
    parser.add_argument('-f', '--infile', type=str, help="for 'bin-decision-slope' only")
    parser.add_argument('-o', '--outdir', type=str)
    parser.add_argument('-g', '--outfile', type=str)
    parser.add_argument('-t', '--task', choices=['fit-decision-slope', 'bin-decision-slope', 'sat-exp-1', 'sat-exp-2'], type=str)
    args = parser.parse_args()
    main(args)
