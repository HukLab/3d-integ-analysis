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
    d1 = pd.Series(d1)
    d2 = pd.Series(d2)
    bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), nbins)
    d1.hist(bins=bins, alpha=0.3, color=clr1)
    d2.hist(bins=bins, alpha=0.3, color=clr2)
    plt.show()

# rss = lambda y, yh: ((y - yh)**2).sum()
# bic_gaussian = lambda y, yh, k: len(y)*np.log(rss(y, yh)) - (len(y)-k)*np.log(n)
bic_gaussian = lambda rss, ny, k: ny*np.log(rss) - (ny-k)*np.log(ny)

def compare_bic(f_pts, f1, f2, k_f1, k_f2):
    df_pts = load_pts(f_pts)
    df_pts = df_pts[df_pts['bi']==0]
    dff1 = load_fits(f1, df_pts)
    dff2 = load_fits(f2, df_pts)
    df1r = residuals(df_pts, dff1)
    df2r = residuals(df_pts, dff2)

    df1r['se'] = df1r['resid']**2
    df2r['se'] = df2r['resid']**2

    sse1 = df1r.groupby('bi_y').sum()['se']#.hist(bins=bins, alpha=0.3)
    sse2 = df2r.groupby('bi_y').sum()['se']#.hist(bins=bins, alpha=0.3)
    # plt.show()
    ny = len(sse1)
    assert len(sse2) == ny
    bic1 = [bic_gaussian(sse, ny, k_f1) for sse in sse1.values]
    bic2 = [bic_gaussian(sse, ny, k_f2) for sse in sse2.values]
    plot_hists(bic1, bic2)
    1/0

def outfilei(outfile, i):
    return outfile.replace('.', '-{0}.'.format(i))

def show(outfile, i):
    print i
    if outfile:
        plt.savefig(outfilei(outfile, i))
    else:
        plt.show()
    return i+1
    
def main(f1, f2, outfile=None, plot_fits=True, plot_hists=True, plot_res=False):
    firstElbow = (136.0, 171.0)
    secondElbow = (983.0, 1267.0)
    I = 0
    df = pd.read_csv(f1)
    dff = pd.read_csv(f2)
    dff0 = dff[dff['bi']==0]
    dff1 = dff.groupby('dotmode', as_index=False).median()
    dff2 = dff.groupby('dotmode', as_index=False).mean()
    dfall = limb(df, dff0) # fit to all bootstrapped sensitivities
    dfmed = limb(df, dff1)
    dfmean = limb(df, dff2)

    if plot_fits:
        plt.figure()
        # df = df[df[KEY] > 0]
        df1 = df.groupby(['dotmode', 'dur'], as_index=False).mean()
        df1[df1['dotmode']=='2d'].plot('dur', KEY, ax=plt.gca(), loglog=True, color='b', kind='scatter')
        df1[df1['dotmode']=='3d'].plot('dur', KEY, ax=plt.gca(), loglog=True, color='r', kind='scatter')
        df1.groupby('dotmode').plot('dur', KEY, ax=plt.gca(), loglog=True, color='black', linestyle='--')
        dfmed[dfmed['dotmode']=='2d'].plot('dur', KEY, ax=plt.gca(), loglog=True, color='b')
        dfmed[dfmed['dotmode']=='3d'].plot('dur', KEY, ax=plt.gca(), loglog=True, color='r')
        # dfmean.groupby('dotmode').plot('dur', KEY, ax=plt.gca(), loglog=True)
        # dfall.groupby('dotmode').plot('dur', KEY, ax=plt.gca(), loglog=True)

        plt.plot([firstElbow[0], firstElbow[0]], [0.1, 1000.0], color='b')
        plt.plot([secondElbow[0], secondElbow[0]], [0.1, 1000.0], color='b')
        plt.plot([firstElbow[1], firstElbow[1]], [0.1, 1000.0], color='r')
        plt.plot([secondElbow[1], secondElbow[1]], [0.1, 1000.0], color='r')
        plt.title('median fit')
        I = show(outfile, I)

    dff0['name'] = 'ALL'
    dff1['name'] = 'MEAN'
    dff2['name'] = 'MEDIAN'
    dffA = dff0.append([dff1, dff2]).sort('dotmode')
    print dffA

    dfres_med = residuals(df, dfmed)
    dfres_mean = residuals(df, dfmean)
    dfres_all = residuals(df, dfall)

    for dm in ['2d', '3d']:
        print (dfres_med[dfres_med['dotmode'] == dm]['resid']**2).sum()
    # 1/0
    if plot_hists:
        for dotmode in ['2d', '3d']:
            dff[dff['dotmode'] == dotmode].hist('m0', bins=50, color='g' if dotmode == '2d' else 'r')
            plt.title('m0: ' + dotmode)
            I = show(outfile, I)
            if 'x0' in dff:
                dff[dff['dotmode'] == dotmode].plot('x0', 'm0', kind='scatter', color='g' if dotmode == '2d' else 'r')
                plt.title('x0, m0: ' + dotmode)
                I = show(outfile, I)
                dff[dff['dotmode'] == dotmode].hist('m1', bins=50, color='g' if dotmode == '2d' else 'r')
                plt.title('m1: ' + dotmode)
                I = show(outfile, I)
                dff[dff['dotmode'] == dotmode].plot('x0', 'm1', kind='scatter', color='g' if dotmode == '2d' else 'r')
                plt.title('x0, m1: ' + dotmode)
                I = show(outfile, I)
                dff[dff['dotmode'] == dotmode].plot('m0', 'm1', kind='scatter', color='g' if dotmode == '2d' else 'r')
                plt.title('m0, m1: ' + dotmode)
                I = show(outfile, I)

    if plot_res:
        for dotmode in ['2d', '3d']:
            dfres_med[dfres_med['dotmode'] == dotmode].plot('dur', 'resid', loglog=False, logx=True, kind='scatter', color='g' if dotmode == '2d' else 'r')
            plt.title('median fit: residuals: ' + dotmode)
            I = show(outfile, I)

        # dfres_med[abs(dfres_med['resid']) < 0.3].groupby('dotmode').plot('dur', 'resid', loglog=False, logx=True, kind='scatter')
        # plt.title('median fit: residuals (less than 0.3)')
        # plt.show()
        # dfres_mean[dfres_mean.dotmode == dm].plot('dur', 'resid', ax=plt.gca(), loglog=False, logx=True, kind='scatter', color='g')
        # dfres_all[dfres_all.dotmode == dm].plot('dur', 'resid', ax=plt.gca(), loglog=False, logx=True, kind='scatter', color='r')
        # plt.ylim([-0.1, 0.5])

if __name__ == '__main__':
    # indir = '../plots/tri-limb-free'
    subjs = ['ALL']#, 'KRM', 'KLB', 'HUK', 'LKC', 'LNK']
    dirname = 'fits0'
    indir = '/Users/mobeets/Desktop/' + dirname
    indir = '../fits'
    outdir = indir + '/figs'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for subj in subjs:
        # indir = '../plots/tmp4'
        # indir = '../plots/twin-limb-zero-drop2'
        # indir = '../plots/twin-limb-zero'
        fn1 = 'pcorVsCohByDur_thresh-{0}-params.csv'.format(subj)
        fn2 = 'pcorVsCohByDur_elbow-{0}-params.csv'.format(subj)
        fn3 = 'pcorVsCohByDur_1elbow-{0}-params.csv'.format(subj)
        gn1 = subj + '.png'
        f1 = os.path.join(indir, fn1)
        f2 = os.path.join(indir, fn2)
        f3 = os.path.join(indir, fn3)
        g1 = os.path.abspath(os.path.join(outdir, gn1))
        compare_bic(f1, f2, f3, 5, 3) # 5,3 are number of free parameters
        # main(f1, f2, g1)
