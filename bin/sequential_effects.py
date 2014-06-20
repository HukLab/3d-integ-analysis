import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from sample import bootstrap, bootstrap_se
from session_info import bad_sessions, all_subjs

def plot(ress, subj, outfile):
    width = 0.5
    fig, ax = plt.subplots()
    cols = ['g', 'r']
    for i, cond in enumerate(ress):
        cases = sorted(ress[cond].keys())
        xs = np.arange(len(cases))
        ms = [ress[cond][case][0]-0.5 for case in cases]
        vs = [ress[cond][case][1] for case in cases]
        plt.bar(xs + i*width/len(ress), ms, label=cond, bottom=[0.5]*len(ms), width=width/len(ress), color=cols[i], yerr=vs)
    plt.plot(xs, [0.5]*len(xs))
    ax.set_xticks(xs + width/2.0)
    ax.set_xticklabels(cases)
    plt.ylim([0.0, 1.0])
    plt.title('Sequential Effects for {0}'.format(subj))
    plt.xlabel('outcome of trial N')
    plt.ylabel('probability of choosing R on trial N+1')
    plt.savefig(outfile)
    plt.show()

def bootstrap_mean_and_se(bools, trials, NBOOTS=1000):
    inds = np.where(bools)[0]
    ts = trials[inds[:-1]+1] # drop last trial
    bss = bootstrap(ts, NBOOTS)
    bs = np.mean(bss, axis=1)
    return np.mean(bs), bootstrap_se(bs)

def sequential_effects(t1chosen, t1correct, sess_ids):
    assert len(t1chosen) == len(t1correct)
    t1chosen = np.array(t1chosen)
    t1correct = np.array(t1correct)
    res = {}

    if sess_ids:
        sess_ids = np.array(sess_ids + (sess_ids[-1],))
        extras = sess_ids[:-1] == sess_ids[1:] # look for session id changes and ignore those trials
        extras = extras + [True]
    else:
        extras = np.array([True]*len(t1chosen))

    # prob R given Right Hit
    bools = t1chosen & t1correct & extras
    res['RH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Right Miss
    bools = t1chosen & ~t1correct & extras
    res['RM'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Hit
    bools = ~t1chosen & t1correct & extras
    res['LH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Miss
    bools = ~t1chosen & ~t1correct & extras
    res['LM'] = bootstrap_mean_and_se(bools, t1chosen)

    return res

def parse(df):
    df = df.sort(['dotmode', 'subj', 'number', 'trial_index'])
    df['resp_is_right'] = df['response'] == 2
    ds = {}
    for dotmode, dfc in df.groupby('dotmode'):
        ds[dotmode] = zip(*dfc[['resp_is_right', 'correct', 'session_index']].values)
    return ds

def load_random():
    N = 10000
    t1chosen = [np.random.binomial(1, 0.5) for _ in xrange(N)]
    t1correct = [np.random.binomial(1, 0.2) for _ in xrange(N)]
    return {'rand': [t1chosen, t1correct, None]}

def main(args, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    OUTDIR = os.path.join(BASEDIR, 'res', 'sequential-effects', outdir)
    outfile = os.path.join(OUTDIR, '{0}-{1}.png'.format(subj, '_'.join(conds)))

    df = load(args)
    ds = parse(df)
    print 'Loaded'
    ress = dict((cond, sequential_effects(*d)) for cond, d in ds.iteritems())
    print 'Calculated'
    plot(ress, subj, outfile)

parser = argparse.ArgumentParser()
parser.add_argument('-c', "--conds", type=str, nargs='*', default=['2d', '3d'], choices=['2d', '3d'], help="condition")
parser.add_argument('-s', "--subj", type=str, choices=all_subjs, help="subject name")
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
args = parser.parse_args()

if __name__ == '__main__':
    main({'subj': args.subj, 'dotmode': args.conds}, args.outdir)
