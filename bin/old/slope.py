import math
import pickle
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from settings import good_subjects

def load_pickle(indir, subj, cond):
    infile = os.path.join(indir, '{0}-{1}-fit.pickle'.format(subj, cond))
    return pickle.load(open(infile))

def plot(res, subj, dotmode, color):
    xs, ys = zip(*res[subj][dotmode])
    plt.scatter(xs, ys, color=color, label='{0}'.format(dotmode))
    zs = np.polyfit(xs, ys, 1)
    xs1 = np.array(list(set(xs)))
    plt.plot(xs1, xs1*zs[0] + zs[1], color=color, linestyle='-')

def main(indir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INDIR = os.path.join(BASEDIR, 'res', indir)

    subj = 'ALL'
    slope = lambda th: (1 - 1/math.e)*(th['A'] - 0.50)/th['T']
    out = {}
    for dotmode in good_subjects.keys():
        for subj in good_subjects[dotmode] + ["ALL"]:
            try:
                res = load_pickle(INDIR, subj, dotmode)['fits']['sat-exp']
            except IOError:
                continue
            if subj not in out:
                out[subj] = {}
            out[subj][dotmode] = []
            for coh, ths in res.iteritems():
                cur = [(coh, slope(th)) for th in ths if slope(th) < 0.1]
                out[subj][dotmode].extend(cur)

    for subj in out:
        plot(out, subj, '2d', 'g')
        plot(out, subj, '3d', 'r')
        plt.title(subj)
        # plt.xscale('log')
        plt.show()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--indir", required=True, type=str, help="The directory from which fits will be loaded.")
    args = parser.parse_args()
    main(args.indir)
