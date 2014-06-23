import os.path
import math

from numpy import random, log10, logspace, exp

from session_info import BINS
from mle import pick_best_theta
from huk_tau_e import huk_tau_e
import saturating_exponential

NTRIALS = 50000
p_correct = lambda dur, (A, B, T): saturating_exponential.saturating_exp(dur, A, B, T)

def uniform_durations(n, (min_dur, max_dur)):
    return random.uniform(min_dur, max_dur, n)

def logspace_durations(n, (min_dur, max_dur)):
    return logspace(log10(min_dur), log10(max_dur), n)

def simulate(ntrials, og_durs, (A, B, T), expdurdist):
    min_dur, max_dur = og_durs[0], og_durs[-1]
    if expdurdist:
        durs = logspace_durations(ntrials, (min_dur, max_dur))
    else:
        durs = uniform_durations(ntrials, (min_dur, max_dur))
    ps = [(dur, p_correct(dur, (A, B, T))) for dur in durs]
    data = [(dur, random.binomial(1, p)) for dur, p in ps]
    return data

def rmse(og_durs, (A, B, T), theta):
    s = 0
    for dur in og_durs:
        y = p_correct(dur, (A, B, T))
        yh = p_correct(dur, (theta['A'], theta['B'], theta['T']))
        s += math.pow((y-yh), 2)
    return math.sqrt(s)

def sim_and_compare_fits(mult=1, quick=False, subj='huk', cond='2d', coh=0.12, A=0.9, B=0.5, T=0.25):
    """
    mult = trial multiple
    quick = take first solution
    """
    ntrials, og_durs = NTRIALS, BINS
    ts = simulate(mult*ntrials, og_durs, (A, B, T), expdurdist=True)
    assert len(ts) == mult*ntrials

    def_dict = {'A': -1.0, 'B': -1.0, 'T': -1.0, 'rmse': -1.0}
    huk, mle_t, mle_a_t, mle_a_b_t = dict(def_dict), dict(def_dict), dict(def_dict), dict(def_dict)
    # print 'ACTUALS: {0}'.format({'A': A, 'B': B, 'T': T})
    # print '----------'

    # print 'HUK'
    thetas_huk, A_huk = huk_tau_e(ts, og_durs)
    if not thetas_huk:
        print 'ERROR: No solutions found'
    else:
        th_huk = pick_best_theta(thetas_huk, True)
        huk = {'A': A_huk, 'B': 0.5, 'T': th_huk[0]}
        huk['rmse'] = rmse(og_durs, (A, B, T), huk)
        # print huk
    # print '----------'

    # print 'MLE: T'
    thetas_mle = saturating_exponential.fit(ts, (A_huk, 0.5, None), quick=quick)
    if not thetas_mle:
        print 'ERROR: No solutions found'
    else:
        th_mle = pick_best_theta(thetas_mle, True)
        mle_t = {'A': A_huk, 'B': 0.5, 'T': th_mle[0]}
        mle_t['rmse'] = rmse(og_durs, (A, B, T), mle_t)
        # print mle_t
    # print '----------'

    # print 'MLE: A, T'
    thetas_mle = saturating_exponential.fit(ts, (None, 0.5, None), quick=quick)
    if not thetas_mle:
        print 'ERROR: No solutions found'
    else:
        th_mle = pick_best_theta(thetas_mle, True)
        mle_a_t = {'A': th_mle[0], 'B': 0.5, 'T': th_mle[1]}
        mle_a_t['rmse'] = rmse(og_durs, (A, B, T), mle_a_t)
        # print mle_a_t
    # print '----------'

    # print 'MLE: A, B, T'
    thetas_mle = saturating_exponential.fit(ts, (None, None, None), quick=quick)
    if not thetas_mle:
        print 'ERROR: No solutions found'
    else:
        th_mle = pick_best_theta(thetas_mle, True)
        mle_a_b_t = {'A': th_mle[0], 'B': th_mle[1], 'T': th_mle[2]}
        mle_a_b_t['rmse'] = rmse(og_durs, (A, B, T), mle_a_b_t)
        # print mle_a_b_t
    # print '----------'

    return {'huk': huk, 'mle_t': mle_t, 'mle_a_t': mle_a_t, 'mle_a_b_t': mle_a_b_t}

def main(outfile, nrepeats=10, nmults=10):
    """
    compare models for various multiples of the original sample size
    """
    rows = []
    for mult in range(1, nmults+1):
        for i in xrange(nrepeats):
            print (mult, i)
            result = sim_and_compare_fits(mult, True)
            for method, res in result.iteritems():
                res['method'] = method
                res['mult'] = mult
                rows.append(res)

    import csv
    fieldnames = ['method', 'mult', 'rmse', 'A', 'B', 'T']
    with open(outfile, 'wb') as csvfile:
        csvwriter = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
        csvwriter.writerow(dict((fn,fn) for fn in fieldnames))
        for row in rows:
            csvwriter.writerow(row)

if __name__ == '__main__':
    main('tmp.csv')
