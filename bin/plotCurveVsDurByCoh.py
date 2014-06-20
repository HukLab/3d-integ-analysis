import pickle
import os.path
import logging
import argparse
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt

from tools import color_list
from sample import bootstrap_se
from fitCurveVsDurByCoh import makefn
from session_info import all_subjs, good_subjects, FIT_IS_COHLESS, LINESTYLE_MAP, COLOR_MAP, MARKER_MAP

from twin_limb import twin_limb
from quick_1974 import quick_1974
from saturating_exponential import saturating_exp
from drift_diffuse import drift_diffusion, drift_diffusion_2

logging.basicConfig(level=logging.DEBUG)

NON_COH_METHODS = [f for f, v in FIT_IS_COHLESS.iteritems() if v]
METHODS = [f for f, v in FIT_IS_COHLESS.iteritems() if not v]

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

def pcor_curves(results, cohs, bins, subj, cond, outfile):
    min_dur, max_dur = min(bins), max(bins)
    # xs = np.linspace(min_dur, max_dur)
    xs = np.logspace(np.log10(min_dur), np.log10(max_dur))
    yf = lambda x, th: saturating_exp(x, th['A'], th['B'], th['T'])
    yfB = lambda x, th: [twin_limb(x, th['X0'], th['S0'], th['P']) for x in xs]
    yf2 = lambda xs, th: [drift_diffusion((C, x), th['K'], th['X0']) for (C, x) in xs]
    yf22 = lambda xs, th: [drift_diffusion_2(x, th['K'], th['X0']) for x in xs]
    yf2B = lambda xs, th: [quick_1974((C, x), th['A'], th['B']) for (C, x) in xs]
    # FIT_FCNS = {'huk': yf, 'sat-exp': yf, 'twin-limb': yfB, 'drift': yf22, 'quick_1974': yf2B}
    FIT_FCNS = {'huk': yf, 'sat-exp': yf, 'twin-limb': yfB, 'drift': yf2, 'quick_1974': yf2B}

    nrows = 3
    ncols = int((len(cohs)-0.5)/nrows)+1
    sec_to_ms = lambda xs: [x*1000 for x in xs]
    color = COLOR_MAP[cond]
    colmap = dict((coh, col) for coh, col in zip([0]*1 + cohs, color_list(len(cohs) + 1, "YlGnBu")))

    plt.clf()
    for i, coh in enumerate(cohs):
        # plt.subplot(ncols, nrows, i+1)
        # plt.title('{0}% coherence'.format(int(coh*100)))
        # plt.xlabel('duration (ms)')
        # plt.ylabel('% correct')
        color = colmap[coh]
        label = int(coh*100)
        for method in results:
            # label = method.upper()
            if method in NON_COH_METHODS:
                for res in results[method]:
                    xs2 = [(coh, x) for x in xs]
                    ys = FIT_FCNS[method](xs2, res)
                    plt.plot(sec_to_ms(xs), ys, color=color, linewidth=2, linestyle=LINESTYLE_MAP[method], label=label)
            elif method in METHODS:
                for res in results[method][coh]:
                    ys = FIT_FCNS[method](xs, res)
                    plt.plot(sec_to_ms(xs), ys, color=color,  linewidth=2, linestyle=LINESTYLE_MAP[method], label=label)
            # plot_fit_hist(results[method], outfile.replace('.png', '{0}_{1}_{2}.png'.format(cond, coh, method)))
            # yserr = pcor_curve_error(results[method], xs2, yf2)
            # plt.errorbar(sec_to_ms(xs), ys, yerr=yserr, fmt=None, ecolor=COLOR_MAP[cond])
        xs_binned, ys_binned = zip(*results['binned_pcor'][coh].iteritems())
        if ys_binned and len(ys_binned[0]) == 2:
            ys_binned, yserr_binned = zip(*ys_binned)
        elif ys_binned and len(ys_binned[0]) == 3:
            ys_binned, yserr_binned, ys_n = zip(*ys_binned)
        else:
            raise Exception("Internal")
        plt.plot(sec_to_ms(xs_binned), ys_binned, color=color, marker='.', linestyle='None')
        plt.errorbar(sec_to_ms(xs_binned), ys_binned, yerr=yserr_binned, fmt=None, ecolor=color)
        # plt.xscale('log')
        # plt.ylim(0.2, 1.1)
    # try:
    #     plt.tight_layout()
    # except:
    #     pass
    plt.title('{0} {1}'.format(subj.upper(), cond))
    plt.xlabel('duration (ms)')
    plt.ylabel('% correct')
    plt.xscale('log')
    plt.xlim(30, 1600)
    plt.ylim(0.4, 1.05)
    # plt.legend()
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    plt.savefig(outfile)

def param_curve(xss, yss, yerrs, colors, markers, linestyles, labels, outfile, title, ylabel):
    plt.clf()
    plt.title(title)
    plt.xlabel('coherence')
    plt.ylabel(ylabel)
    assert len(xss) == len(yss) == len(colors) == len(labels) == len(markers) == len(linestyles)
    for xs, ys, yerr, col, mkr, lin, lbl in zip(xss, yss, yerrs, colors, markers, linestyles, labels):
        plt.plot(xs, ys, color=col, linestyle=lin, marker=mkr, label=lbl, markersize=4)
        plt.errorbar(xs, ys, yerr=yerr, fmt=None, ecolor=col)
    plt.xscale('log')
    # plt.yscale('log')
    if yss:
        plt.ylim(0, max(chain(*yss)))
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

def param_curve_both_conds(results, cohs, methods, outfile, key, title, ylabel):
    """ results keyed first by cond """
    xss = []
    yss = []
    yerrs = []
    cols = []
    mkrs = []
    lins = []
    lbls = []
    
    def get_param_bounds(r, k):
        v = bootstrap_se([x[k] for x in r])
        return v, v

    for cond, res in results.iteritems():
        for method in methods:
            if method not in res:
                continue
            xs = sorted(res[method].keys()) # cohs
            xss.append(xs)
            if method in NON_COH_METHODS:
                ys = [coh*np.mean([x[key] for x in res[method]]) for coh in xs]
                # ys = [res[method][0][key]*coh for coh in xs]
                yerr = [[coh*x for x in get_param_bounds(res[method], key)] for coh in xs]
            else:
                # for coh in xs:
                #     plot_fit_hist(res[method][coh], outfile.replace('.png', '_' + cond + '_' + str(coh) + '.png'))
                ys = [np.mean([x[key] for x in res[method][coh]]) for coh in xs]
                # ys = [res[method][coh][0][key] for coh in xs]
                yerr = [get_param_bounds(res[method][coh], key) for coh in xs]
            yss.append(ys)
            yerrs.append(zip(*yerr)) # [lower_bounds, upper_bounds]
            lbl = '{0}-{1}'.format(cond, method)
            cols.append(COLOR_MAP[cond])
            mkrs.append(MARKER_MAP[cond])
            lins.append(LINESTYLE_MAP[method])
            lbls.append(lbl)
    if xss:
        param_curve(xss, yss, yerrs, cols, mkrs, lins, lbls, outfile, title, ylabel)

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
    for subj in subjs:
        subj_conds = [cond for cond in conds if subj in subjs_by_cond[cond]]
        results_both, res = {}, {}
        for cond in subj_conds:
            try:
                res = load_pickle(INDIR, subj, cond)
            except IOError:
                msg = "could not load {0} {1}".format(subj, cond)
                logging.error(msg)
                continue
            outfile = makefn(OUTDIR, subj, cond, 'fit', 'png')
            pcor_curves(res['fits'], res['cohs'], res['bins'], subj, cond, outfile)
            results_both[cond] = res['fits']
        if res:
            param_curve_both_conds(results_both, res['cohs'], ['drift'], makefn(OUTDIR, subj, 'cond', 'K'), 'K', 'K per coherence', 'K*coh', 'png')
            param_curve_both_conds(results_both, res['cohs'], ['twin-limb'], makefn(OUTDIR, subj, 'cond', 't0'), 'X0', 'Elbow time per coherence', 't0', 'png')
            methods = ['sat-exp', 'huk']
            param_curve_both_conds(results_both, res['cohs'], methods, makefn(OUTDIR, subj, 'cond', 'A'), 'A', 'Saturation % correct per coherence', 'A', 'png')
            param_curve_both_conds(results_both, res['cohs'], methods, makefn(OUTDIR, subj, 'cond', 'tau'), 'T', 'Time constants per coherence', 'tau (ms)', 'png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str, help="The directory from which fits will be loaded.")
    parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
    parser.add_argument('-c', "--conds", default=['2d', '3d'], nargs='*', choices=['2d', '3d'], type=str, help="The number of imperatives to generate.")
    parser.add_argument('-s', "--subj", default='SUBJECT', type=str, choices=['SUBJECT', 'ALL'] + all_subjs, help="SUBJECT fits for each subject, ALL combines data and fits all at once.")
    args = parser.parse_args()
    main(args.conds, args.subj, args.indir, args.outdir)
