import json
import os.path
import argparse
import numpy as np
import pandas as pd
import pypsignifit as psi

from pd_io import load
from sample import bootstrap
from pmf_elbows import find_elbows_per_boots
from pmf_plot import plot_logistics, plot_threshes, make_durmap, label_fcn, Finv

THRESH_VAL = 0.75

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
    pts, (theta, thresh), booted = threshold_and_bootstrap(dfc, thresh_val, nboots)
    print 'theta={0}, thresh={1}'.format(theta, thresh)
    if nboots > 0:
        return pts, (theta, thresh), booted
    return pts, (theta, thresh), []

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

def main(ps, nbins, nboots, ignore_dur, doPlot, outdir, isLongDur, nElbows):
    df = load(ps, None, 'both' if isLongDur else False, nbins)
    durmap = make_durmap(df)
    rows1, rows2 = [], []
    for dotmode, df_dotmode in df.groupby('dotmode'):
        if ignore_dur:
            r1, r2 = make_rows(df_dotmode, dotmode, durmap, di, nboots)
            rows1.extend(r1)
            rows2.extend(r2)
        else:
            for di, df_durind in df_dotmode.groupby('duration_index'):
                r1, r2 = make_rows(df_durind, dotmode, durmap, di, nboots)
                rows1.extend(r1)
                rows2.extend(r2)

    df_pts = pd.DataFrame(rows1, columns=['subj', 'dotmode', 'di', 'dur', 'x', 'y', 'ntrials'])
    df_fts = pd.DataFrame(rows2, columns=['subj', 'dotmode', 'di', 'dur', 'bi', 'thresh', 'loc', 'scale', 'lapse'])
    df_fts['subj'] = ps['subj']
    df_fts['dur'] = 1000*df_fts['dur']
    
    if not ignore_dur and nElbows > 0:
        df_elbs = find_elbows_per_boots(df_fts, nElbows)
    else:
        df_elbs = pd.DataFrame()
    if doPlot and not ignore_dur:
        plot_logistics(df_pts, df_fts)
        plot_threshes(df_fts, df_elbs)
    if outdir is not None:
        pass
        # to_json(df, nbins, nboots, res, df_elbs, outdir, ignore_dur)

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
