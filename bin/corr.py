import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pd_io import load, resample_by_grp

def regress(df, doPlot):
    vals = {}
    for dotmode, dfd in df.groupby('dotmode'):
        vals[dotmode] = []
        for durind, dfc in dfd.groupby('duration_index'):
            dfc1 = dfc.groupby('coherence', as_index=False)['correct'].agg([np.mean, len])['correct'].reset_index()
            X, y, _ = zip(*dfc1[['coherence','mean','len']].values)
            X = sm.tools.add_constant(X)
            fit = sm.OLS(y, X).fit()
            val = fit.rsquared
            vals[dotmode].append((durind, val))
        if doPlot:
            plt.scatter(*zip(*vals[dotmode]), color='g' if dotmode == '2d' else 'r')
    if doPlot:
        plt.show()
    return vals

def check_different(xss):
    nboots, nbins = xss.shape
    m0 = np.mean(np.mean(xss, 1))
    s0 = np.mean(np.std(xss, 1))
    lb0, ub0 = m0 - s0, m0 + s0
    ms, ss = np.mean(xss, 0), np.std(xss, 0, ddof=1)
    lb, ub = ms - ss, ms + ss
    msk = (ub < lb0)
    return np.arange(nbins)[msk], ub[msk]

def main(isLongDur=True, nbins=20, resample=5, nboots=1000, doPlot=False):
    df0 = load({}, None, 'both' if isLongDur else False, nbins)
    xss = np.zeros([nboots, nbins])
    yss = np.zeros([nboots, nbins])
    for i in xrange(nboots):
        if resample:
            df = resample_by_grp(df0, resample)
        vals = regress(df, doPlot)
        xs = np.array(vals['2d'])[:,1]
        ys = np.array(vals['3d'])[:,1]
        xss[i, :] = xs
        yss[i, :] = ys
    durinds_to_ignore_2d, ubs_2d = check_different(xss)
    durinds_to_ignore_3d, ubs_3d = check_different(yss)
    print ('2d', zip(durinds_to_ignore_2d, ubs_2d)), ('3d', zip(durinds_to_ignore_3d, ubs_3d))

if __name__ == '__main__':
    main()
