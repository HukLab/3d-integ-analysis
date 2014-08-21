import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pmf_plot import make_durmap

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

# def find_elbows(df, res, nElbows):
#     durmap = make_durmap(df)
#     elbs = {}
#     for dotmode, res1 in res.iteritems():
#         elbs[dotmode] = {}
#         pts = []
#         for di, res0 in res1.iteritems():
#             pts.extend([(durmap[di], thresh) for theta, thresh in res0['fit']])
#         if not pts:
#             return
#         xs, ys = zip(*pts)
#         xs = 1000*np.array(xs)
#         if nElbows == 1:
#             th = find_elbow(xs, ys)
#         else:
#             if dotmode == '3d':
#                 print 'WARNING: Ignoring all 3D thresholds at x={0}'.format(xs[0])
#                 xst = list(xs)
#                 assert sorted(xst) == xst
#                 last_instance_index = len(xst) - list(reversed(xst)).index(xst[0]) - 1
#                 xs = xs[last_instance_index+1:]
#                 ys = ys[last_instance_index+1:]
#             th = find_two_elbows(xs, ys)
#         elbs[dotmode]['fit'] = th
#         elbs[dotmode]['binned'] = (xs, ys)
#         print 'elbow={0}'.format(th)
#     return elbs

def find_elbows_one_boot(df, nElbows):
    xs, ys = zip(*df[['dur', 'thresh']].values)
    if nElbows == 1:
        th = find_elbow(xs, ys)
    else:
        th = find_two_elbows(xs, ys)
    print 'elbow={0}'.format(th)
    return th, (xs, ys)

def df_res(df, res):
    durmap = make_durmap(df)
    rows = []
    for dotmode, res1 in res.iteritems():
        for di, res0 in res1.iteritems():
            rows.extend([(dotmode, bi, durmap[di], thresh) for bi, (theta, thresh) in enumerate(res0['fit'])])
    df = pd.DataFrame(rows, columns=['dotmode', 'bi', 'dur', 'thresh'])
    df['dur'] = 1000*df['dur']
    return df

def find_elbows_per_boots(df, res, nElbows):
    elbs = {}
    dfr = df_res(df, res)
    for dotmode, dfp in dfr.groupby('dotmode'):
        ths = []
        Pts = []
        elbs[dotmode] = {}
        if dotmode == '3d':
            print 'WARNING: Ignoring all 3D thresholds at x={0}'.format(dfp['dur'].min())
            dfp = dfp[dfp['dur'] > dfp['dur'].min()]
        for bi, dfpts in dfp.groupby('bi'):
            th, pts = find_elbows_one_boot(dfpts, nElbows)
            ths.append(th)
            Pts.append(pts)
        elbs[dotmode]['fit'] = ths
        elbs[dotmode]['binned'] = zip(*dfp[['dur', 'thresh']].values)
    return elbs
