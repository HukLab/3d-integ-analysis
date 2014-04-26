import pickle
import os.path
import logging
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt

from dio import makefn
from fcns import pc_per_dur_by_coh
from session_info import good_subjects

logging.basicConfig(level=logging.DEBUG)

METHODS = ['huk', 'mle']
COL_MAP = {'2d': 'g', '3d': 'r'}
MKR_MAP = {'2d': 'o', '3d': 'o'}
LIN_MAP = {'huk': 'dashed', 'mle': 'solid'}

def pcor_curves(results, bins, cond, outfile):
    min_dur, max_dur = min(bins), max(bins)
    xs = np.linspace(min_dur, max_dur)
    yf = lambda x, th: pc_per_dur_by_coh(x, th['A'], th['B'], th['T'])

    cohs = sorted(results.keys())
    nrows = 3
    ncols = int((len(cohs)-0.5)/nrows)+1
    sec_to_ms = lambda xs: [x*1000 for x in xs]

    plt.clf()
    for i, coh in enumerate(cohs):
        plt.subplot(ncols, nrows, i+1)
        plt.title('{0}% coherence'.format(int(coh*100)))
        plt.xlabel('duration (ms)')
        plt.ylabel('% correct')
        ys_mle = yf(xs, results[coh]['mle'])
        ys_huk = yf(xs, results[coh]['huk'])
        xs_binned, ys_binned = zip(*results[coh]['binned'].iteritems())
        plt.plot(sec_to_ms(xs), ys_mle, color=COL_MAP[cond], linestyle='solid', label='MLE')
        plt.plot(sec_to_ms(xs), ys_huk, color=COL_MAP[cond], linestyle='dashed', label='HUK')
        plt.plot(sec_to_ms(xs_binned), ys_binned, 'o')
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

def tau_curve(xss, yss, colors, markers, linestyles, labels, outfile):
    plt.clf()
    plt.title('Time constants per coherence')
    plt.xlabel('coherence')
    plt.ylabel('tau (ms)')
    assert len(xss) == len(yss) == len(colors) == len(labels) == len(markers) == len(linestyles)
    for xs, ys, col, mkr, lin, lbl in zip(xss, yss, colors, markers, linestyles, labels):
        plt.plot(xs, ys, color=col, linestyle=lin, marker=mkr, label=lbl)
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

def tau_curve_both_fits(results, cond, outfile, method=None):
    if method:
        methods = [method]
    else:
        methods = METHODS
    xss = []
    yss = []
    cols = []
    mkrs = []
    lins = []
    lbls = []
    get_tau = lambda r, coh, method: r[coh][method]['T']
    for method in methods:
        xs = sorted(results.keys())
        xss.append(xs)
        yss.append([get_tau(results, coh, method) for coh in xs])
        cols.append(COL_MAP[cond])
        mkrs.append(MKR_MAP[cond])
        lins.append(LIN_MAP[method])
        lbls.append(method)
    tau_curve(xss, yss, cols, mkrs, lins, lbls, outfile)

def tau_curve_both_conds(results, outfile, method=None):
    """ results keyed first by cond """
    assert len(results.keys()) == 2
    if method:
        methods = [method]
    else:
        methods = METHODS
    xss = []
    yss = []
    cols = []
    mkrs = []
    lins = []
    lbls = []
    get_tau = lambda r, coh, method: r[coh][method]['T']
    for cond, res in results.iteritems():
        for method in methods:
            xs = sorted(res.keys())
            xss.append(xs)
            yss.append([get_tau(res, coh, method) for coh in xs])
            lbl = '{0}-{1}'.format(cond, method)
            cols.append(COL_MAP[cond])
            mkrs.append(MKR_MAP[cond])
            lins.append(LIN_MAP[method])
            lbls.append(lbl)
    tau_curve(xss, yss, cols, mkrs, lins, lbls, outfile)

def plot_curves(fits, bins, cond, outfiles):
    pcor_curves(fits, bins, cond, outfiles['fit'])
    tau_curve_both_fits(fits, cond, outfiles['tau'])

def make_outfiles(outdir, subj, cond):
    return {'fit': makefn(outdir, subj, cond, 'fit', 'png'), 'tau': makefn(outdir, subj, cond, 'tau', 'png')}

def load_pickle(indir, subj, cond):
    infile = os.path.join(indir, '{0}-{1}-fit.pickle'.format(subj, cond))
    return pickle.load(open(infile))

def main(conds, subjs_by_cond, indir, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INDIR = os.path.join(BASEDIR, 'res', indir)
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)            
    
    all_subjs = list(set(chain(*subjs_by_cond.values())))
    for subj in all_subjs:
        subj_conds = [cond for cond in conds if subj in subjs_by_cond[cond]]
        results_both = {}
        for cond in subj_conds:
            res = load_pickle(INDIR, subj, cond)
            outfiles = make_outfiles(OUTDIR, subj, cond)
            plot_curves(res['fits'], res['bins'], cond, outfiles)
            results_both[cond] = res['fits']
        if len(subj_conds) > 1:
            outfile = makefn(OUTDIR, subj, 'both', 'tau', 'png')
            tau_curve_both_conds(results_both, outfile)

if __name__ == '__main__':
    conds = ['2d', '3d']
    subjs = dict((cond, good_subjects[cond]) for cond in conds)
    # subjs = dict((cond, ['ALL']) for cond in conds)
    indir = 'fits'
    outdir = 'plots'
    main(conds, subjs, indir, outdir)
