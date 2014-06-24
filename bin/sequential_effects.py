import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from sample import bootstrap, bootstrap_se
from settings import bad_sessions, all_subjs

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
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()

def bootstrap_mean_and_se(bools, trials, NBOOTS=1000):
    inds = np.where(bools)[0]
    ts = trials[inds[:-1]+1] # drop last trial
    bss = bootstrap(ts, NBOOTS)
    bs = np.mean(bss, axis=1)
    return np.mean(bs), bootstrap_se(bs)

def make_mask(sess_ids, tr_ids):
    assert len(sess_ids) == len(tr_ids)
    sess_ids = np.array(sess_ids)
    tr_ids = np.array(tr_ids)

    # look for session id changes and ignore those trials
    sess_ids = np.concatenate([sess_ids, [sess_ids[-1]]])
    m1 = sess_ids[:-1] == sess_ids[1:]

    # look for trial skips and ignore those trials
    tr_ids = np.concatenate([tr_ids, [tr_ids[-1] + 1]])
    m2 = tr_ids[:-1] == tr_ids[1:] - 1

    return m1 & m2

def sequential_effects(t1chosen, t1correct, mask):
    assert len(t1chosen) == len(t1correct)
    t1chosen = np.array(t1chosen)
    t1correct = np.array(t1correct)
    res = {}

    # prob R given Right Hit
    bools = t1chosen & t1correct & mask
    res['RH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Right Miss
    bools = t1chosen & ~t1correct & mask
    res['RM'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Hit
    bools = ~t1chosen & t1correct & mask
    res['LH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Miss
    bools = ~t1chosen & ~t1correct & mask
    res['LM'] = bootstrap_mean_and_se(bools, t1chosen)

    return res

def calculate(ds):
    res = {}
    for dotmode, (a, b, c, d) in ds.iteritems():
        mask = make_mask(c, d) # ignores session changes, trial skips
        res[dotmode] = sequential_effects(a, b, mask)
    return res

def load_random():
    N = 10000
    t1chosen = [np.random.binomial(1, 0.5) for _ in xrange(N)]
    t1correct = [np.random.binomial(1, 0.2) for _ in xrange(N)]
    return {'rand': [t1chosen, t1correct, None]}

def parse(df):
    df = df.sort(['dotmode', 'subj', 'number', 'trial_index'])
    df['resp_is_right'] = df['response'] == 2
    ds = {}
    for dotmode, dfc in df.groupby('dotmode'):
        ds[dotmode] = zip(*dfc[['resp_is_right', 'correct', 'session_index', 'trial_index']].values)
    return ds

def main(args, outdir):
    if outdir:
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
        OUTDIR = os.path.join(BASEDIR, 'res', 'sequential-effects', outdir)
        outfile = os.path.join(OUTDIR, '{0}-{1}.png'.format(subj, '_'.join(conds)))
    else:
        outfile = None

    df = load(args)
    ds = parse(df)
    print 'Loaded'
    res = calculate(ds)
    print 'Calculated'
    subjs = df['subj'].unique()
    plot(res, subjs[0] if len(subjs) == 1 else 'ALL', outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dotmode", type=str, choices=['2d', '3d'], help="condition")
    parser.add_argument('-s', "--subj", type=str, choices=all_subjs, help="subject name")
    parser.add_argument('-o', "--outdir", type=str, help="The directory to which fits will be written.")
    args = parser.parse_args()
    main({'subj': args.subj, 'dotmode': args.dotmode}, args.outdir)
