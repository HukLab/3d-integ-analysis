import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pmf_plot import make_durmap

is_nan_or_inf = lambda items: np.isnan(items) | np.isinf(items)
def find_elbow(xs, ys, ntries=10):
    x0min = None
    x0max = max(xs)/2.0
    x0min = np.log(x0min) if x0min else x0min
    x0max = np.log(x0max) if x0max else x0max

    zs = np.array([np.log(xs), np.log(ys)])
    xs, ys = zs[:, ~is_nan_or_inf(zs[0]) & ~is_nan_or_inf(zs[1])]
    def error_fcn((x0, A0, B0, A1, B1)):
        z = np.array([xs < x0, xs*A0 + B0, xs*A1 + B1])
        yh = z[0]*z[1] + (1-z[0])*z[2]
        return np.sum(np.power(ys-yh, 2))
    bounds = [(x0min, x0max), (None, 0), (None, None), (None, 0), (None, None)]
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
    x1max = max(xs) - 200
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
    bounds = [(x0min, x0max), (None, APPROX_ZERO), (None, None), (None, APPROX_ZERO), (None, None), (x1min, x1max), (-APPROX_ZERO, APPROX_ZERO), (None, None)]
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

def find_elbows_one_boot(df, nElbows):
    xs, ys = zip(*df[['dur', 'thresh']].values)
    if nElbows == 1:
        th = find_elbow(xs, ys)
        keys = ['x0', 'm0', 'b0', 'm1', 'b1']
    else:
        th = find_two_elbows(xs, ys)
        keys = ['x0', 'm0', 'b0', 'm1', 'b1', 'x1', 'm2', 'b2']
    print 'elbow={0}'.format(th)
    return dict(zip(keys, th)) if th is not None else {}

def find_elbows_per_boots(dfr, nElbows):
    rows = []
    for dotmode, dfp in dfr.groupby('dotmode'):
        if dotmode == '3d':
            print 'WARNING: Ignoring all 3D thresholds at x={0}'.format(dfp['dur'].min())
            dfp = dfp[dfp['dur'] > dfp['dur'].min()]
        for bi, dfpts in dfp.groupby('bi'):
            row = find_elbows_one_boot(dfpts, nElbows)
            row.update({'dotmode': dotmode, 'bi': bi})
            rows.append(row)
    return pd.DataFrame(rows)
