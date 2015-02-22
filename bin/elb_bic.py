import os.path
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt

KEY = 'sens' # 'thresh'
def one_limb(df, dfm):
    dfp = df[['di', 'dur','dotmode']].drop_duplicates().reset_index()[['di','dur','dotmode']]
    dfc = dfp.merge(dfm)
    dfc[KEY] = dfc['dur']**dfc['m0']*np.exp(dfc['b0'])
    return dfc

def twin_limb(df, dfm):
    dfp = df[['di', 'dur','dotmode']].drop_duplicates().reset_index()[['di','dur','dotmode']]
    dfc = dfp.merge(dfm)

    dfc['dur0'] = np.exp(dfc['x0'])
    dfc['d0'] = dfc['dur'] <= dfc['dur0']
    dfc['d1'] = dfc['dur'] > dfc['dur0']

    assert dfc[['d0','d1']].sum().sum() == len(dfc)

    dfc['y0'] = dfc['dur']**dfc['m0']*np.exp(dfc['b0'])
    dfc['y1'] = dfc['dur']**dfc['m1']*np.exp(dfc['b1'])

    dfc[KEY] = dfc['d0']*dfc['y0'] + dfc['d1']*dfc['y1']
    return dfc

def tri_limb(df, dfm):
    dfp = df[['di', 'dur','dotmode']].drop_duplicates().reset_index()[['di','dur','dotmode']]

    # xs = np.linspace(min(df['dur']), max(df['dur']))
    # dm = ['2d', '3d']
    # dfp = pd.DataFrame(list(product(dm, xs)), columns=['dotmode', 'dur'])

    dfc = dfp.merge(dfm)

    dfc['dur0'] = np.exp(dfc['x0'])
    dfc['dur1'] = np.exp(dfc['x1'])
    dfc['d0'] = dfc['dur'] <= dfc['dur0']
    dfc['d1'] = (dfc['dur'] > dfc['dur0']) & (dfc['dur'] < dfc['dur1'])
    dfc['d2'] = dfc['dur'] >= dfc['dur1']
    assert dfc[['d0','d1','d2']].sum().sum() == len(dfc)

    dfc['y0'] = dfc['dur']**dfc['m0']*np.exp(dfc['b0'])
    dfc['y1'] = dfc['dur']**dfc['m1']*np.exp(dfc['b1'])
    dfc['y2'] = dfc['dur']**dfc['m2']*np.exp(dfc['b2'])

    dfc[KEY] = dfc['d0']*dfc['y0'] + dfc['d1']*dfc['y1'] + dfc['d2']*dfc['y2']
    return dfc

def limb(df1, df2):
    if 'm2' in df2:
        return tri_limb(df1, df2)
    elif 'm1' in df2:
        return twin_limb(df1, df2)
    else:
        return one_limb(df1, df2)

def residuals(df, dff):
    inds = ['dotmode', 'dur']
    df1 = df.groupby(inds, as_index=False).mean()
    df2 = df1.merge(dff, on=inds)
    df2['resid'] = df2[KEY + '_x'] - df2[KEY + '_y']
    return df2

def load_pts(infile):
    df = pd.read_csv(infile)
    df = df[(df['sens'] >= 0.0)]# & (df['sens'] < 150.0)]
    # remove first two duration bins for 3d
    inds = (df['dotmode']=='3d') & (df['di'] < 3)
    df = df[~inds]
    return df

def load_fits(infile, df_pts):
    dff = pd.read_csv(infile)
    df1 = limb(df_pts, dff)
    return df1

def plot_hists(d1, d2, nbins=100, clr1='k', clr2='g'):
    bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), nbins)
    d1.hist(bins=bins, alpha=0.3, color=clr1)
    d2.hist(bins=bins, alpha=0.3, color=clr2)
    plt.show()

bic_gaussian = lambda rss, ny, k: ny*np.log(rss) - (ny-k)*np.log(ny)
lbf = lambda pct: (1.0-pct)/2.0

def ci(df, pct=0.682): #0.955
    qs = [lbf(pct), 1-lbf(pct)]
    dff = df.quantile(qs + [0.5])
    return dff.values

def compare_bic(f_pts, (f1, k_f1, lbl_f1), (f2, k_f2, lbl_f2)):
    df_pts = load_pts(f_pts)
    df_pts = df_pts[df_pts['bi']==0]
    dff1 = load_fits(f1, df_pts)
    dff2 = load_fits(f2, df_pts)
    df1r = residuals(df_pts, dff1)
    df2r = residuals(df_pts, dff2)

    df1r['se'] = df1r['resid']**2
    df2r['se'] = df2r['resid']**2

    sse1 = df1r.groupby('bi_y').sum()['se']
    sse2 = df2r.groupby('bi_y').sum()['se']

    ny = len(sse1)
    assert len(sse2) == ny
    bic1 = pd.Series([bic_gaussian(sse, ny, k_f1) for sse in sse1.values])
    bic2 = pd.Series([bic_gaussian(sse, ny, k_f2) for sse in sse2.values])
    lb, ub, med = ci(bic1)
    print '{0}: {1}    C.I.=({2}, {3})'.format(lbl_f1, med, lb, ub)
    lb, ub, med = ci(bic2)
    print '{0}: {1}    C.I.=({2}, {3})'.format(lbl_f2, med, lb, ub)
    plot_hists(bic1, bic2)

if __name__ == '__main__':
    subj = 'ALL'
    indir = '../fits'
    fn1 = 'pcorVsCohByDur_thresh-{0}-params.csv'.format(subj)
    fn2 = 'pcorVsCohByDur_elbow-{0}-params.csv'.format(subj)
    fn3 = 'pcorVsCohByDur_1elbow-{0}-params.csv'.format(subj)
    f1 = os.path.join(indir, fn1)
    f2 = os.path.join(indir, fn2)
    f3 = os.path.join(indir, fn3)
    compare_bic(f1, (f2, 5, 'tri-limb'), (f3, 3, 'bi-limb'))
