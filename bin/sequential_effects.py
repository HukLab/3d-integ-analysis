import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from dio import load_json
from sample import bootstrap, bootstrap_se
from summaries import session_grouper, group_trials

def plot(res, (subj, cond)):
    cases = sorted(res.keys())
    xs = np.arange(len(cases))
    ms = [res[case][0]-0.5 for case in cases]
    vs = [res[case][1] for case in cases]
    width = 0.5

    fig, ax = plt.subplots()
    plt.bar(xs, ms, bottom=[0.5]*len(ms), width=width, color='r', yerr=vs)
    plt.plot(xs, [0.5]*len(xs))
    ax.set_xticks(xs + width/2.0)
    ax.set_xticklabels(cases)
    plt.ylim([0.0, 1.0])
    plt.title('Sequential Effects in {0} for {1}'.format(cond, subj))
    plt.xlabel('outcome of trial N')
    plt.ylabel('probability of choosing R on trial N+1')
    plt.show()

def bootstrap_mean_and_se(bools, trials, NBOOTS=1000):
    inds = np.where(bools)[0]
    ts = trials[inds[:-1]+1] # drop last trial
    bss = bootstrap(ts, NBOOTS)
    bs = np.mean(bss, axis=1)
    return np.mean(bs), bootstrap_se(bs)

def sequential_effects(t1chosen, t1correct):
    assert len(t1chosen) == len(t1correct)
    t1chosen = np.array(t1chosen)
    t1correct = np.array(t1correct)
    res = {}

    # prob R given Right Hit
    bools = np.logical_and(t1chosen==1, t1correct==1)
    res['R_RH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Right Miss
    bools = np.logical_and(t1chosen==1, t1correct==0)
    res['R_RM'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Hit
    bools = np.logical_and(t1chosen==0, t1correct==1)
    res['R_LH'] = bootstrap_mean_and_se(bools, t1chosen)

    # prob R given Left Miss
    bools = np.logical_and(t1chosen==0, t1correct==0)
    res['R_LM'] = bootstrap_mean_and_se(bools, t1chosen)

    return res

def load(subj, cond):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    TRIALS = load_json(INFILE)
    groups = group_trials(TRIALS, session_grouper, False)
    trials = groups[(subj, cond)]
    return zip(*[(t.response == 2, t.correct) for t in trials])

def load_random():
    N = 10000
    t1chosen = [np.random.binomial(1, 0.5) for _ in xrange(N)]
    t1correct = [np.random.binomial(1, 0.2) for _ in xrange(N)]
    return t1chosen, t1correct

def main(subj, cond):
    t1chosen, t1correct = load(subj, cond)
    # t1chosen, t1correct = load_random()
    print 'Loaded'
    res = sequential_effects(t1chosen, t1correct)
    print 'Calculated'
    plot(res, (subj, cond))

parser = argparse.ArgumentParser()
parser.add_argument('-c', "--cond", type=str, choices=['2d', '3d'], help="condition")
parser.add_argument('-s', "--subj", type=str, help="subject name")
args = parser.parse_args()

if __name__ == '__main__':
    main(args.subj, args.cond)
