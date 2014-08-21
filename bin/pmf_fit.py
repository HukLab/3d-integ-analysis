import json
import os.path
import argparse
import numpy as np
import pandas as pd
import pypsignifit as psi
from scipy.optimize import minimize

from pd_io import load
from sample import bootstrap
from pmf_plot import plot_logistics, plot_threshes, make_durmap, label_fcn, Finv

THRESH_VAL = 0.75

is_nan_or_inf = lambda items: np.isnan(items) | np.isinf(items)
def find_elbow(xs, ys, ntries=10):
    zs = np.array([np.log(xs), np.log(ys)])
    xs, ys = zs[:, ~is_nan_or_inf(zs[0]) & ~is_nan_or_inf(zs[1])]
    def error_fcn((x0, A0, B0, A1, B1)):
        z = np.array([xs < x0, xs*A0 + B0, xs*A1 + B1])
        yh = z[0]*z[1] + (1-z[0])*z[2]
        return np.sum(np.power(ys-yh, 2))
    bounds = [(min(xs), max(xs)), (None, 0), (None, None), (None, 0), (None, None)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.array([x[0]*(x[1] - x[3]) + x[2] - x[4]]) }]
    guess = np.array([np.mean(xs), -1, 0, -0.5, 0])
    for i in xrange(ntries):
        soln = minimize(error_fcn, guess*(1 + i/10.), method='SLSQP', bounds=bounds, constraints=constraints)
        if soln['success']:
            return soln['x']
    return None

def find_two_elbows(xs, ys, ntries=10):
    x0min = None #min(xs)
    x0max = max(xs)*1/3.0 # first 2000 ms
    x1min = None #max(xs)*1/4.0 #  last 2000 ms
    x1max = max(xs) - 100
    print 'NOTE: ENFORCING ELBOW BOUNDS'
    print 'min={0}, max={1}, x0=({2}, {3}), x1=({4}, {5})'.format(min(xs), max(xs), x0min, x0max, x1min, x1max)

    APPROX_ZERO = 0.0001
    zs = np.array([np.log(xs), np.log(ys)])
    xs, ys = zs[:, ~is_nan_or_inf(zs[0]) & ~is_nan_or_inf(zs[1])]
    x0min = np.log(x0min) if x0min else x0min
    x0max = np.log(x0max) if x0max else x0max
    x1min = np.log(x1min) if x1min else x1min
    x1max = np.log(x1max) if x1max else x1max

    def error_fcn((x0, A0, B0, A1, B1, x1, A2, B2)):
        z = np.array([xs < x0, xs*A0 + B0, xs*A1 + B1, xs > x1, xs*A2 + B2])
        yh = z[0]*z[1] + z[3]*z[4] + (1-z[0])*(1-z[3])*z[2]
        return (ys-yh).dot(ys-yh) # np.sum(np.power(ys-yh, 2))
    bounds = [(x0min, x0max), (None, APPROX_ZERO), (None, None), (None, APPROX_ZERO), (None, None), (x1min, x1max), (None, APPROX_ZERO), (None, None)]
    # bounds = [(min(xs), max(xs)), (None, APPROX_ZERO), (None, None), (None, APPROX_ZERO), (None, None), (min(xs), max(xs)), (None, APPROX_ZERO), (None, None)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.array([x[0]*(x[1] - x[3]) + x[2] - x[4]]) }]
    constraints.append({'type': 'eq', 'fun': lambda x: np.array([x[5]*(x[6] - x[3]) + x[7] - x[4]]) })
    constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[5] - x[0]]) })
    guess = np.array([np.mean(xs), -1, 0.0, -0.5, 0.0, np.mean(xs)+0.5, 0.0, 0.0])
    for i in xrange(ntries):
        soln = minimize(error_fcn, guess*(1 + i/10.), method='SLSQP', bounds=bounds, constraints=constraints)
        if soln['success']:
            th = soln['x']
            print 'x0={0}'.format(th[0])
            print 'x1={0}'.format(th[5])
            print 'm0={0}, b0={1}'.format(th[1], th[2])
            print 'm1={0}, b1={1}'.format(th[3], th[4])
            print 'm2={0}, b2={1}'.format(th[6], th[7])
            return soln['x']
    return None

def find_elbows(df, res, nElbows):
    durmap = make_durmap(df)
    elbs = {}
    for dotmode, res1 in res.iteritems():
        elbs[dotmode] = {}
        pts = []
        for di, res0 in res1.iteritems():
            pts.extend([(durmap[di], thresh) for theta, thresh in res0['fit']])
        if not pts:
            return
        xs, ys = zip(*pts)
        xs = 1000*np.array(xs)
        if nElbows == 1:
            th = find_elbow(xs, ys)
        else:
            if dotmode == '3d':
                print 'WARNING: Ignoring all 3D thresholds at x={0}'.format(xs[0])
                xst = list(xs)
                assert sorted(xst) == xst
                last_instance_index = len(xst) - list(reversed(xst)).index(xst[0]) - 1
                xs = xs[last_instance_index+1:]
                ys = ys[last_instance_index+1:]
            th = find_two_elbows(xs, ys)
        elbs[dotmode]['fit'] = th
        elbs[dotmode]['binned'] = (xs, ys)
        print 'elbow={0}'.format(th)
    return elbs

def find_theta_and_thresh(xs, ys, zs, thresh_val, nboots=0):
    data = np.array(zip(xs, ys, zs))
    a = "Gauss(0,5)"
    b = "Gauss(1,3)"
    l = "Beta(1.5,12)"
    B = psi.BootstrapInference(data, priors=[a, b, l], nafc=2, sample=nboots)
    thetas = B.mcestimates if B.mcestimates is not None else [B.estimate]
    threshes = [Finv(thresh_val, th) for th in thetas]
    mean_theta = np.mean(thetas, 0)
    mean_thresh = Finv(thresh_val, mean_theta)
    return mean_theta, mean_thresh, thetas, threshes

def make_data(dfc):
    dfc1 = dfc.groupby('coherence', as_index=False)['correct'].agg([np.mean, len])['correct'].reset_index()
    xs, ys, zs = zip(*dfc1[['coherence','mean','len']].values)
    xs = np.array(xs)
    ys = np.array(ys).astype('float')
    zs = np.array(zs)
    return xs, ys, zs

def threshold_and_bootstrap(dfc, thresh_val, nboots):
    xs, ys, zs = make_data(dfc)
    mean_theta, mean_thresh, thetas, threshes = find_theta_and_thresh(xs, ys, zs, thresh_val, nboots)
    return (xs, ys, zs), (mean_theta, mean_thresh), zip(thetas, threshes)

def threshold(dfc, nboots, thresh_val=THRESH_VAL):
    out = {}
    pts, (theta, thresh), booted = threshold_and_bootstrap(dfc, thresh_val, nboots)
    print 'theta={0}, thresh={1}'.format(theta, thresh)
    out['fit'] = [(theta, thresh)]
    out['binned'] = pts
    if nboots > 0:
        out['fit'].extend(booted)
    return out

def unique_fname(filename):
    if not os.path.exists(filename):
        return filename
    i = 1
    update_ofcn = lambda infile, i: infile.replace('.json', '-{0}.json'.format(i))
    while os.path.exists(update_ofcn(filename, i)):
        i += 1
    return update_ofcn(filename, i)

def to_json(df, nbins, nboots, res, elbs, outdir, ignore_dur):
    class NumPyArangeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist() # or map(int, obj)
            return json.JSONEncoder.default(self, obj)
            
    subjs = df['subj'].unique()
    subj = '{0}'.format(subjs[0].upper()) if len(subjs) == 1 else 'ALL'
    ofcn = lambda label, subj, dotmode: 'pcorVsCohByDur_{label}-{subj}-{dotmode}-{nbins}.json'.format(label=label, subj=subj, dotmode=dotmode, nbins=nbins)
    # ofcn = lambda label, subj, dotmode: 'pcorVsCohByDur_{label}-{subj}-{dotmode}.json'.format(label=label, subj=subj, dotmode=dotmode)
    durmap = make_durmap(df)
    for dotmode, res1 in res.iteritems():
        obj = [{'di': di, 'dur': durmap[di], 'obj': res0} for di, res0 in res1.iteritems()]
        json_outfile = os.path.join(outdir, ofcn('thresh' if not ignore_dur else 'thresh_by_dotmode', subj, dotmode))
        with open(unique_fname(json_outfile), 'w') as f:
            json.dump(obj, f, cls=NumPyArangeEncoder, indent=4)
    for dotmode, obj in elbs.iteritems():
        json_outfile = os.path.join(outdir, ofcn('elbow', subj, dotmode))
        with open(unique_fname(json_outfile), 'w') as f:
            json.dump(obj, f, cls=NumPyArangeEncoder, indent=4)

def main(ps, nbins, nboots, ignore_dur, doPlot, outdir, isLongDur, nElbows):
    df = load(ps, None, 'both' if isLongDur else False, nbins)
    durmap = make_durmap(df)
    res = {}
    for dotmode, df_dotmode in df.groupby('dotmode'):
        if ignore_dur:
            print '{0}'.format(dotmode)
            res[dotmode] = {}
            res[dotmode][0] = threshold(df_dotmode, nboots)
        else:
            res[dotmode] = {}
            for di, df_durind in df_dotmode.groupby('duration_index'):
                print '{0}, di={1}, d={2}ms, n={3}'.format(dotmode, di, label_fcn(1000*durmap[di]), len(df_durind))
                res[dotmode][di] = threshold(df_durind, nboots)
    if not ignore_dur and nElbows > 0:
        elbs = find_elbows(df, res, nElbows)
    else:
        elbs = {}
    if doPlot and not ignore_dur:
        # plot_logistics(df, res)
        plot_threshes(df, res, elbs)
    if outdir is not None:
        to_json(df, nbins, nboots, res, elbs, outdir, ignore_dur)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    parser.add_argument('--durind', required=False, type=int)
    parser.add_argument('-b', '--nboots', required=False, type=int, default=0)
    parser.add_argument('-n', '--nbins', required=False, type=int, default=None)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--ignore-dur', action='store_true', default=False)
    parser.add_argument('-e', '--n-elbows', type=int, default=1)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode, 'duration_index': args.durind}
    main(ps, args.nbins, args.nboots, args.ignore_dur, args.plot, args.outdir, args.is_long_dur, args.n_elbows)
