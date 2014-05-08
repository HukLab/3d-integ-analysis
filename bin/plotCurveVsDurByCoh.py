import pickle
import os.path
import logging
import argparse
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt

from dio import makefn
from saturating_exponential import saturating_exp
from drift_diffuse import drift_diffusion
from session_info import all_subjs, good_subjects

logging.basicConfig(level=logging.DEBUG)

METHODS = ['huk', 'sat-exp']
NON_COH_METHODS = ['drift']
COL_MAP = {'2d': 'g', '3d': 'r'}
MKR_MAP = {'2d': 's', '3d': 's'}
LIN_MAP = {'huk': 'dashed', 'sat-exp': 'solid', 'drift': 'dotted'}

def pcor_curve_error(fits, xs, ys_fcn):
    """
    fits is list of bootstrapped fits, each of which is a dict keyed by params
    ys_fcn returns the y-value evaluated at xs with a fit from fits
    returns the lower- and upper-bound of ys_fcn at a given x in xs across all fits
    """
    yss = [ys_fcn(xs, fit) for fit in fits]
    ys_low, ys_high = [], []
    for i, x in enumerate(xs):
        ysi = [ys[i] for ys in yss]
        ys_low.append(min(ysi))
        ys_high.append(max(ysi))
    return [np.array(ys_low)-yss[0], np.array(ys_high)-yss[0]]

def plot_fit_hist(fits, outfile):
    keys = fits[0].keys()
    mean = dict((k, np.mean([fit[k] for fit in fits])) for k in keys)
    var = dict((k, np.var([fit[k] for fit in fits])) for k in keys)
    plt.clf()
    for i, k in enumerate(keys):
        plt.subplot(len(keys), 1, i+1)
        plt.title(k)
        plt.hist([fit[k] for fit in fits])
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(outfile)

def pcor_curves(results, cohs, bins, cond, outfile):
    min_dur, max_dur = min(bins), max(bins)
    # xs = np.linspace(min_dur, max_dur)
    xs = np.logspace(np.log10(min_dur), np.log10(max_dur))
    yf = lambda x, th: saturating_exp(x, th['A'], th['B'], th['T'])
    yf2 = lambda xs, th: [drift_diffusion((C, x), th['K']) for (C, x) in xs]

    nrows = 3
    ncols = int((len(cohs)-0.5)/nrows)+1
    sec_to_ms = lambda xs: [x*1000 for x in xs]

    plt.clf()
    for i, coh in enumerate(cohs):
        plt.subplot(ncols, nrows, i+1)
        plt.title('{0}% coherence'.format(int(coh*100)))
        plt.xlabel('duration (ms)')
        plt.ylabel('% correct')
        for method in METHODS:
            if method not in results:
                continue
            # plot_fit_hist(results[method][coh], outfile.replace('.png', '{0}_{1}_{2}.png'.format(cond, coh, method)))
            ys = yf(xs, results[method][coh][0])
            plt.plot(sec_to_ms(xs), ys, color=COL_MAP[cond], linestyle=LIN_MAP[method], label=method.upper())
            for res in results[method][coh]:
                ys = yf(xs, res)
                plt.plot(sec_to_ms(xs), ys, color=COL_MAP[cond], linestyle=LIN_MAP[method], label=method.upper())
            # yserr = pcor_curve_error(results[method][coh], xs, yf)
            # plt.errorbar(sec_to_ms(xs), ys, yerr=yserr, fmt=None, ecolor=COL_MAP[cond])
        for method in NON_COH_METHODS:
            if method not in results:
                continue
            # plot_fit_hist(results[method], outfile.replace('.png', '{0}_{1}_{2}.png'.format(cond, coh, method)))
            xs2 = [(coh, x) for x in xs]
            ys = yf2(xs2, results[method][0])
            plt.plot(sec_to_ms(xs), ys, color=COL_MAP[cond], linestyle=LIN_MAP[method], label=method.upper())
            for res in results[method]:
                ys = yf2(xs2, res)
                plt.plot(sec_to_ms(xs), ys, color=COL_MAP[cond], linestyle=LIN_MAP[method], label=method.upper())
            # yserr = pcor_curve_error(results[method], xs2, yf2)
            # plt.errorbar(sec_to_ms(xs), ys, yerr=yserr, fmt=None, ecolor=COL_MAP[cond])
        xs_binned, ys_binned = zip(*results['binned_pcor'][coh].iteritems())
        plt.plot(sec_to_ms(xs_binned), ys_binned, color=COL_MAP[cond], marker='o', linestyle='None')
        plt.xscale('log')
        # plt.xlim()
        plt.ylim(0.2, 1.1)
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

def tau_curve(xss, yss, yerrs, colors, markers, linestyles, labels, outfile):
    plt.clf()
    plt.title('Time constants per coherence')
    plt.xlabel('coherence')
    plt.ylabel('tau (ms)')
    assert len(xss) == len(yss) == len(colors) == len(labels) == len(markers) == len(linestyles)
    for xs, ys, yerr, col, mkr, lin, lbl in zip(xss, yss, yerrs, colors, markers, linestyles, labels):
        plt.plot(xs, ys, color=col, linestyle=lin, marker=mkr, label=lbl)
        plt.errorbar(xs, ys, yerr=yerr, fmt=None, ecolor=col)
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(0, 1200)
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

def tau_curve_both_conds(results, cohs, outfile, method=None):
    """ results keyed first by cond """
    if method:
        methods = [method]
    else:
        methods = METHODS
    xss = []
    yss = []
    yerrs = []
    cols = []
    mkrs = []
    lins = []
    lbls = []
    
    def get_tau_bounds(r, k):
        vals = [x[k] for x in r]
        v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0 # ddof=1 => divide by N-1
        return v, v

    for cond, res in results.iteritems():
        for method in methods:
            if method not in res:
                continue
            xs = [coh for coh in cohs if coh in res[method]]
            xss.append(xs)
            if method in NON_COH_METHODS:
                ys = [res[method][0]['K']*coh for coh in xs]
                yerr = [get_tau_bounds(res[method], 'K')*coh for coh in xs]
            else:
                ys = [res[method][coh][0]['T'] for coh in xs]
                yerr = [get_tau_bounds(res[method][coh], 'T') for coh in xs]
            yss.append(ys)
            yerrs.append(zip(*yerr)) # [lower_bounds, upper_bounds]
            lbl = '{0}-{1}'.format(cond, method)
            cols.append(COL_MAP[cond])
            mkrs.append(MKR_MAP[cond])
            lins.append(LIN_MAP[method])
            lbls.append(lbl)
    tau_curve(xss, yss, yerrs, cols, mkrs, lins, lbls, outfile)

def make_outfiles(outdir, subj, cond):
    return {'fit': makefn(outdir, subj, cond, 'fit', 'png'), 'tau': makefn(outdir, subj, cond, 'tau', 'png')}

def load_pickle(indir, subj, cond):
    infile = os.path.join(indir, '{0}-{1}-fit.pickle'.format(subj, cond))
    return pickle.load(open(infile))

def main(conds, subj, indir, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INDIR = os.path.join(BASEDIR, 'res', indir)
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)

    if subj == 'SUBJECT':
        subjs_by_cond = dict((cond, good_subjects[cond]) for cond in conds)
    elif subj == 'ALL':
        subjs_by_cond = dict((cond, [subj]) for cond in conds)
    elif subj in all_subjs:
        subjs_by_cond = dict((cond, [subj]) for cond in conds)
    else:
        msg = "subj {0} not recognized".format(subj)
        logging.error(msg)
        raise Exception(msg)
    
    subjs = list(set(chain(*subjs_by_cond.values())))
    cohs = []
    for subj in subjs:
        subj_conds = [cond for cond in conds if subj in subjs_by_cond[cond]]
        results_both = {}
        for cond in subj_conds:
            res = load_pickle(INDIR, subj, cond)
            outfiles = make_outfiles(OUTDIR, subj, cond)
            if not cohs:
                cohs = res['cohs']
            pcor_curves(res['fits'], res['cohs'], res['bins'], cond, outfiles['fit'])
            results_both[cond] = res['fits']
        outfiles = make_outfiles(OUTDIR, subj, 'cond')
        tau_curve_both_conds(results_both, cohs, outfiles['tau'])

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--indir", required=True, type=str, help="The directory from which fits will be loaded.")
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
parser.add_argument('-c', "--conds", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="The number of imperatives to generate.")
parser.add_argument('-s', "--subj", default='SUBJECT', type=str, choices=['SUBJECT', 'ALL'] + all_subjs, help="SUBJECT fits for each subject, ALL combines data and fits all at once.")
args = parser.parse_args()

if __name__ == '__main__':
    main(args.conds, args.subj, args.indir, args.outdir)
