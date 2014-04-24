import csv
import json
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

from dio import load_json
from fcns import pc_per_dur_by_coh
from session_info import BINS, good_subjects, bad_sessions, good_cohs, bad_cohs
from fit_compare import pick_best_theta
from sample import sample_wr
from summaries import group_trials, subj_grouper, dot_grouper, session_grouper, coherence_grouper, as_x_y
from huk_tau_e import binned_ps, huk_tau_e
from mle_set_B import mle_set_B
from mle import mle

logging.basicConfig(level=logging.DEBUG)

def write_fit_json(results, bins, outfile):
    out = {}
    out['results'] = results
    out['duration_bins'] = bins
    json.dump(out, open(outfile, 'w'))

def write_fit_csv(results, bins, outfile):
    with open(outfile, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        th_keys = ['A', 'B', 'T']
        header = ['coh'] + th_keys + bins

        cohs = sorted(results.keys())
        header = ['cohs:'] + cohs
        csvwriter.writerow(header)

        # write thetas per coh
        for key in th_keys:
            row = [key]
            for coh in cohs:
                val = results[coh]['mle'][key]
                row.append(val)
            csvwriter.writerow(row)

        # write binned ps per coh
        for dur in bins:
            row = [dur]
            for coh in cohs:
                d = results[coh]['ps']
                if dur in d:
                    val = d[dur]
                else:
                    val = 'N/A'
                row.append(val)
            csvwriter.writerow(row)

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
    # plt.legend()
    try:
        plt.tight_layout()
    except:
        pass
    # plt.show()
    plt.savefig(outfile)

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
        xs_binned, ys_binned = zip(*results[coh]['ps'].iteritems())
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

def mle_fit(ts, B, bins, coh, quick=True):
    ths = mle_set_B(ts, B=B, quick=quick)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found with fixed B={0}. Letting B vary.'.format(B)
        # logging.info(msg)
        # ths = mle(ts, quick=True)
        if not ths:
            msg = 'No fits found. Using alpha=1.0, tau=0.0001'
            logging.warning(msg)
            th = [1.0, 0.001]
    msg = '{0}% MLE: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': th[0], 'B': B if len(th) == 2 else th[1], 'T': th[-1]}

def huk_fit(ts, B, bins, coh):
    ths, A = huk_tau_e(ts, B=B, durs=bins)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found for Huk Using tau=0.0001.'
        logging.warning(msg)
        th = 0.001
    msg = '{0}% HUK: {1}'.format(int(coh*100), th)
    logging.info(msg)
    return {'A': A, 'B': B, 'T': th}

def fit_curves(trials, bins, B=0.5):
    groups = group_trials(trials, coherence_grouper, False)
    cohs = sorted(groups)
    results = {}
    for coh in cohs:
        ts = groups[coh]
        ts_cur_coh = as_x_y(ts)
        logging.info('{0}%: Found {1} trials'.format(int(coh*100), len(ts_cur_coh)))
        results[coh] = {}
        results[coh]['mle'] = mle_fit(ts_cur_coh, B, bins, coh)
        results[coh]['huk'] = huk_fit(ts_cur_coh, B, bins, coh)
        results[coh]['ps'] = binned_ps(ts_cur_coh, bins)
        results[coh]['ntrials'] = len(ts_cur_coh)
    return results

def curves_for_session(trials, bins, subj, cond, outfiles):
    msg = 'Loaded {0} trials for subject {1} and {2} dots'.format(len(trials), subj, cond)
    logging.info(msg)
    if not trials:
        logging.info('No graphs.')
        return
    results = fit_curves(trials, bins)
    make_curves(results, bins, outfiles['fit'])
    tau_curve(results, bins, outfiles['tau'])
    # write_fit_json(results, bins, outfiles['json'])

def outfiles(outdir, subj, cond):
    makefn = lambda name, ext: os.path.join(outdir, '{0}-{1}-{2}.{3}'.format(subj, cond, name, ext))
    return {'fit': makefn('fit', 'png'), 'tau': makefn('tau', 'png'), 'json': makefn('fit', 'json')}

def remove_bad_trials_by_session(trials, cond):
    """ assumes filtering already done by cond """
    subjs = good_subjects[cond]
    bad_sess = bad_sessions[cond]
    keep = lambda t: t.session.subject in subjs and t.session.index not in bad_sess
    return [t for t in trials if keep(t)]

def sample_trials_by_session(trials, cond, mult=5):
    """ assumes filtering already done by cond """
    groups = group_trials(trials, subj_grouper, False)
    n = min(len(ts) for ts in groups.values())
    ts_all = []
    for subj, ts in groups.iteritems():
        ts_cur = sample_wr(ts, mult*n)
        ts_all.extend(ts_cur)
    msg = 'Created {0} trials for {1} dots (sampling with replacement per subject)'.format(len(ts_all), cond)
    logging.info(msg)
    return ts_all

def remove_trials_by_coherence(trials, cond):
    bad = False
    if bad:
        cohs = bad_cohs[cond]
        return [t for t in trials if t.coherence not in cohs]
    else:
        cohs = good_cohs[cond]
        return [t for t in trials if t.coherence in cohs]

def make_curves(subj, cond, trials, bins, outdir):
    trials = remove_bad_trials_by_session(trials, cond)
    trials = remove_trials_by_coherence(trials, cond)
    trials = sample_trials_by_session(trials, cond)
    curves_for_session(trials, bins, subj, cond, outfiles(outdir, subj, cond))

def by_subject(trials, conds, bins, outdir):
    groups = group_trials(trials, session_grouper, False)
    for cond in conds:
        for subj in good_subjects[cond]:
            trials = groups[(subj, cond)]
            make_curves(subj, cond, trials, bins, outdir)

def across_subjects(trials, conds, bins, outdir):
    groups = group_trials(trials, dot_grouper, False)
    subj = 'ALL'
    for cond in conds:
        trials = groups[cond]
        make_curves(subj, cond, trials, bins, outdir)

def main(outdir, kind, conds, bins):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    OUTDIR = os.path.join(BASEDIR, 'res', outdirname)
    TRIALS = load_json(INFILE)
    if kind == 'ALL':
        across_subjects(TRIALS, conds, bins, OUTDIR)
    elif kind == 'SUBJECT':
        by_subject(TRIALS, conds, bins, OUTDIR)
    else:
        msg = "kind {0} not recognized".format(kind)
        logging.error(msg)
        raise Exception(msg)

if __name__ == '__main__':
    conds = ['2d'] # ['2d', '3d']
    outdir = 'BY_SUBJ_2'
    kind = 'SUBJECT'
    # outdir = 'ALL_GOOD_2'
    # kind = 'ALL'
    main(outdir, kind, conds, BINS)

"""
 NO FITS:
    klb 2d: 100%
    krm 2d: 50%, 100%
    lkc 3d: 100%
    huk 3d: 0% --> fixed by run with TNC solver

* MLE: bad at curves with lots of p~1
* MLE: shouldn't always take first guess
* B=0.5 not always true
    * sometimes there is obviously a delay, i.e. 0.5 extends up until some duration t'

TO DO:
    * finer bins for all
    * fit taus with line
    * 2d and 3d on same plot
    * re-gen for ALL
    * re-gen for SUBJ
"""
