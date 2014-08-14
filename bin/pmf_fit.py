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
    x0_max_ms = 0
    bounds = [(min(xs), max(xs)), (None, 0), (None, None), (None, 0), (None, None)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.array([x[0]*(x[1] - x[3]) + x[2] - x[4]]) }]
    guess = np.array([np.mean(xs), -1, 0, -0.5, 0])
    for i in xrange(ntries):
        soln = minimize(error_fcn, guess*(1 + i/10.), method='SLSQP', bounds=bounds, constraints=constraints)
        if soln['success']:
            return soln['x']
    return None

def find_elbows(df, res):
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
        th = find_elbow(xs, ys)
        elbs[dotmode]['fit'] = th
        elbs[dotmode]['binned'] = (xs, ys)
        print 'elbow={0}'.format(th)
    return elbs

def find_theta_and_thresh(xs, ys, zs, thresh_val):
    data = np.array(zip(xs, ys, zs))
    # a = "unconstrained"
    a = "Gauss(0,5)"
    # a = "Gauss(0,100)"
    # b = "unconstrained"
    b = "Gauss(1,3)"
    # b = "Gauss(1,2000)"
    # l = "Beta(2,20)"
    l = "Beta(1.5,12)"
    B = psi.BootstrapInference(data, priors=[a, b, l], nafc=2)
    # B.sample(50)
    # psi.GoodnessOfFit(B)
    # return B.estimate, B.getThres(thresh_val*(1-0.5-B.estimate[-1]) + 0.5)#2*thresh_val-1)
    return B.estimate, Finv(thresh_val, B.estimate)

def make_data(dfc):
    dfc1 = dfc.groupby('coherence', as_index=False)['correct'].agg([np.mean, len])['correct'].reset_index()
    xs, ys, zs = zip(*dfc1[['coherence','mean','len']].values)
    xs = np.array(xs)
    ys = np.array(ys).astype('float')
    zs = np.array(zs)
    return xs, ys, zs

def threshold_bootstrap(dfc, nboots, thresh_val):
    xy = dfc[['coherence','correct']].sort('coherence').groupby('coherence')
    zss = [bootstrap(y.values, nboots) for x,y in xy]
    zss = [np.vstack([z[i] for z in zss]) for i in xrange(nboots)]
    res = []
    for i in xrange(nboots):
        xs = zss[i][:, 0].astype('float')
        ys = zss[i][:, 1].astype('float')
        dfb = pd.DataFrame({'coherence': xs, 'correct': ys, 'session_index': [1]*len(xs)})
        xs, ys, zs = make_data(dfb)
        theta, thresh = find_theta_and_thresh(xs, ys, zs, thresh_val)
        res.append((theta, thresh))
    return res

def threshold_once(dfc, thresh_val):
    xs, ys, zs = make_data(dfc)
    return find_theta_and_thresh(xs, ys, zs, thresh_val), (xs, ys, zs)

def threshold(dfc, nboots, thresh_val=THRESH_VAL):
    out = {}
    (theta, thresh), pts = threshold_once(dfc, thresh_val)
    out['fit'] = [(theta, thresh)]
    out['binned'] = pts
    print 'theta={0}, thresh={1}'.format(theta, thresh)
    if nboots > 0:
        out['fit'].extend(threshold_bootstrap(dfc, nboots, thresh_val))
    return out

def to_json(df, res, elbs, outdir, ignore_dur):
    class NumPyArangeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist() # or map(int, obj)
            return json.JSONEncoder.default(self, obj)
            
    subjs = df['subj'].unique()
    subj = '{0}'.format(subjs[0].upper()) if len(subjs) == 1 else 'ALL'
    ofcn = lambda label, subj, dotmode: 'pcorVsCohByDur_{label}-{subj}-{dotmode}.json'.format(label=label, subj=subj, dotmode=dotmode)
    durmap = make_durmap(df)
    for dotmode, res1 in res.iteritems():
        obj = [{'di': di, 'dur': durmap[di], 'obj': res0} for di, res0 in res1.iteritems()]
        json_outfile = os.path.join(outdir, ofcn('thresh' if not ignore_dur else 'thresh_by_dotmode', subj, dotmode))
        with open(json_outfile, 'w') as f:
            json.dump(obj, f, cls=NumPyArangeEncoder, indent=4)
    for dotmode, obj in elbs.iteritems():
        json_outfile = os.path.join(outdir, ofcn('elbow', subj, dotmode))
        with open(json_outfile, 'w') as f:
            json.dump(obj, f, cls=NumPyArangeEncoder, indent=4)

def main(ps, nboots, ignore_dur, doPlot, outdir, isLongDur):
    df = load(ps, None, 'longDur' if isLongDur else False)
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
    if not ignore_dur:
        elbs = find_elbows(df, res)
    else:
        elbs = {}
    if doPlot and not ignore_dur:
        # plot_logistics(df, res)
        plot_threshes(df, res, elbs)
    if outdir is not None:
        to_json(df, res, elbs, outdir, ignore_dur)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    parser.add_argument('--durind', required=False, type=int)
    parser.add_argument('--nboots', required=False, type=int, default=0)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--ignore-dur', action='store_true', default=False)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode, 'duration_index': args.durind}
    main(ps, args.nboots, args.ignore_dur, args.plot, args.outdir, args.is_long_dur)
