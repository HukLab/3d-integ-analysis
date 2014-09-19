import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pd_io import load, resample_by_grp

def main(isLongDur=True, nbins=20, resample=5):
    df = load({}, None, 'both' if isLongDur else False, nbins)
    if resample:
        df = resample_by_grp(df, resample)

    vals = []
    for dotmode, dfd in df.groupby('dotmode'):
        for durind, dfc in dfd.groupby('duration_index'):
            y = dfc['correct'].astype(int)
            X = dfc['coherence']
            fit = sm.Logit(y, X).fit()
            # val = fit.llr
            # val = fit.prsquared
            val = fit.aic
            # val = fit.cov_params()['coherence'][0]
            # val = fit.params[0]
            vals.append((durind, val))
            # print fit.summary()
            # 1/0
        plt.scatter(*zip(*vals), color='g' if dotmode == '2d' else 'r')
    plt.show()

if __name__ == '__main__':
    main()
