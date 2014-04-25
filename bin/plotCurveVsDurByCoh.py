import pickle
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

from dio import makefn
from fcns import pc_per_dur_by_coh
from session_info import good_subjects

logging.basicConfig(level=logging.DEBUG)

def make_curves(results, bins, outfile):
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
        plt.plot(sec_to_ms(xs), ys_mle, 'r-', label='MLE')
        plt.plot(sec_to_ms(xs), ys_huk, 'g-', label='HUK')
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

def tau_curve(results, bins, outfile):
    get_tau = lambda coh, name: results[coh][name]['T']
    taus_mle = [get_tau(coh, 'mle') for coh in results]
    taus_huk = [get_tau(coh, 'huk') for coh in results]
    plt.clf()
    plt.title('Time constants per coherence')
    plt.xlabel('coherence')
    plt.ylabel('tau (ms)')
    xs = results.keys()
    plt.plot(xs, taus_mle, 'ro', label='MLE')
    plt.plot(xs, taus_huk, 'go', label='HUK')
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

def plot_curves(fits, bins, outfiles):
    make_curves(fits, bins, outfiles['fit'])
    tau_curve(fits, bins, outfiles['tau'])

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
    for cond in conds:
        for subj in subjs_by_cond[cond]:
            res = load_pickle(INDIR, subj, cond)
            outfiles = make_outfiles(OUTDIR, subj, cond)
            plot_curves(res['fits'], res['bins'], outfiles)

if __name__ == '__main__':
    conds = ['2d', '3d']
    subjs = dict((cond, good_subjects[cond]) for cond in conds)
    indir = 'fits'
    outdir = 'plots'
    main(conds, subjs, indir, outdir)
