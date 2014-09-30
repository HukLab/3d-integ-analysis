import os.path
import argparse
import numpy as np
import pandas as pd
import pypsignifit as psi

from pd_io import load, resample_by_grp
from pmf_elbows import find_elbows_per_boots
from pmf_plot import plot_logistics, plot_threshes, make_durmap, label_fcn, Finv

THRESH_VAL = 0.75

def find_theta_and_thresh(xs, ys, zs, thresh_val, nboots=0):
    data = np.array(zip(xs, ys, zs))
    a = "Gauss(0,5)"
    b = "Gauss(1,3)"
    # l = "Beta(2,20)"
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
    pts, (theta, thresh), booted = threshold_and_bootstrap(dfc, thresh_val, nboots)
    print 'theta={0}, thresh={1}'.format(theta, thresh)
    if nboots > 0:
        return pts, (theta, thresh), booted
    return pts, (theta, thresh), []

def make_rows(df, dotmode, durmap, di, nboots):
    print '{0}, di={1}, d={2}ms, n={3}'.format(dotmode, di, label_fcn(1000*durmap[di]), len(df))
    rows1 = []
    rows2 = []
    (xs, ys, zs), mean_fit, boot_fits = threshold(df, nboots)
    # T0 = ['subj', 'dotmode', 'di', 'dur', 'x', 'y', 'ntrials']
    # T1 = ['subj', 'dotmode', 'di', 'dur', 'bi', 'thresh', 'loc', 'scale', 'lapse']
    for (x,y,z) in zip(xs, ys, zs):
        row = {'x': x, 'y': y, 'ntrials': z}
        row.update({'dotmode': dotmode, 'di': di, 'dur': durmap[di]})
        rows1.append(row)
    fits = boot_fits + [list(mean_fit)]
    for bi, (theta, thresh) in enumerate(fits):
        row = {'bi': bi, 'thresh': thresh, 'loc': theta[0], 'scale': theta[1], 'lapse': theta[2]}
        row.update({'dotmode': dotmode, 'di': di, 'dur': durmap[di]})
        rows2.append(row)
    return rows1, rows2

def unique_fname(filename):
    if not os.path.exists(filename):
        return filename
    i = 1
    update_ofcn = lambda infile, i: infile.replace('.csv', '-{0}.csv'.format(i))
    while os.path.exists(update_ofcn(filename, i)):
        i += 1
    return update_ofcn(filename, i)

def to_csv(nbins, nboots, subj, df_pts, df_fts, df_elbs, outdir, ignore_dur):
    key = 'pcorVsCohByDur' if not ignore_dur else 'pcorVsCohByDotmode'
    ofcn2 = lambda label, extra: '{key}_{label}-{subj}-{extra}.csv'.format(key=key, label=label, subj=subj.upper(), extra=extra)
    ofcn1 = lambda label, extra: os.path.join(outdir, ofcn2(label, extra))
    ofcn0 = lambda label, extra: unique_fname(ofcn1(label, extra))
    of1 = ofcn0('thresh', 'pts')
    of2 = ofcn0('thresh', 'params')
    df_pts.to_csv(of1)
    df_fts.to_csv(of2)
    if not ignore_dur:
        of3 = ofcn0('elbow', 'params')
        df_elbs.to_csv(of3)

def main(ps, nbins, nboots, ignore_dur, doPlotPmf, doPlotElb, outdir, isLongDur, nElbows, min_di, enforceZeroSlope, resample=None):
    secondElbow = (983.0, 1894.0)
    df = load(ps, None, 'both' if isLongDur else False, nbins, secondElbow)
    if resample:
        df = resample_by_grp(df, resample)
    durmap = make_durmap(df)
    rows1, rows2 = [], []
    for dotmode, df_dotmode in df.groupby('dotmode'):
        if ignore_dur:
            r1, r2 = make_rows(df_dotmode, dotmode, durmap, 0, nboots)
            rows1.extend(r1)
            rows2.extend(r2)
        else:
            for di, df_durind in df_dotmode.groupby('duration_index'):
                r1, r2 = make_rows(df_durind, dotmode, durmap, di, nboots)
                rows1.extend(r1)
                rows2.extend(r2)

    df_pts = pd.DataFrame(rows1, columns=['subj', 'dotmode', 'di', 'dur', 'x', 'y', 'ntrials'])
    df_fts = pd.DataFrame(rows2, columns=['subj', 'dotmode', 'di', 'dur', 'bi', 'thresh', 'loc', 'scale', 'lapse'])
    df_fts['dur'] = 1000*df_fts['dur']
    df_fts['sens'] = 1/df_fts['thresh']

    if not ignore_dur and nElbows >= 0:
        df_elbs = find_elbows_per_boots(df_fts, nElbows, min_di, enforceZeroSlope)
    else:
        df_elbs = pd.DataFrame()
    if doPlotPmf:
        plot_logistics(df_pts, df_fts)
    if doPlotElb and not ignore_dur:
        plot_threshes(df_fts, df_elbs)
    if outdir is not None:
        subj = ps['subj'] if ps['subj'] is not None else 'ALL'
        df_pts['subj'] = subj
        df_fts['subj'] = subj
        df_elbs['subj'] = subj
        to_csv(nbins, nboots, subj, df_pts, df_fts, df_elbs, outdir, ignore_dur)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    parser.add_argument('--durind', required=False, type=int)
    parser.add_argument('--sessind', required=False, type=int)
    parser.add_argument('-b', '--nboots', required=False, type=int, default=0)
    parser.add_argument('-n', '--nbins', required=False, type=int, default=20)
    parser.add_argument('--plot-pmf', action='store_true', default=False)
    parser.add_argument('--plot-elb', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--ignore-dur', action='store_true', default=False)
    parser.add_argument('-e', '--n-elbows', type=int, default=1)
    parser.add_argument('-r', '--resample', type=int, default=0)
    parser.add_argument('-l', '--is-long-dur', action='store_true', default=False)
    parser.add_argument('--min-di', type=int, default=0)
    parser.add_argument('--enforce-zero', action='store_true', default=True)
    args = parser.parse_args()
    ps = {'subj': args.subj, 'dotmode': args.dotmode, 'duration_index': args.durind, 'session_index': args.sessind}
    main(ps, args.nbins, args.nboots, args.ignore_dur, args.plot_pmf, args.plot_elb, args.outdir, args.is_long_dur, args.n_elbows, args.min_di, args.enforce_zero, args.resample)
