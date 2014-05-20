import os.path
import logging
import argparse

from dio import load_json
from summaries import group_trials, session_grouper, coherence_grouper, as_x_y

logging.basicConfig(level=logging.DEBUG)

METHOD = 'sat-exp'

def load_trials(trials, subj, cond):
    groups = group_trials(trials, session_grouper, False)
    return groups[(subj, cond)]

def load_pickle(indir, subj, cond):
    infile = os.path.join(indir, '{0}-{1}-fit.pickle'.format(subj, cond))
    return pickle.load(open(infile))

def mix(trials, indir, outdir, subj):
    ress = {}
    for cond in ['2d', '3d']:
        ress[cond] = load_pickle(indir, subj, cond)
        # for coh, th in res[METHOD].iteritems():
        #     pass

    # assert As in each cond are sorted for increasing coherences
    cohs = sorted(ress['2d'].keys())
    As2d = [ress['2d'][coh]['A'] for coh in cohs]
    assert As2d == sorted(As2d)
    As3d = [ress['3d'][coh]['A'] for coh in cohs]
    assert As3d == sorted(As3d)

    # find 2d As, check 3d As...
    minA = max(min(As2d), min(As3d))
    maxA = min(max(As2d), max(As3d))
    nAs = len([x for x in As2d + As3d if minA <= a <= maxA])
    goalAs = np.linspace(minA, maxA, nAs)
    msg = 'goal As: {0}'.format(goalAs)
    logging.info(msg)

    # for 2d/3d trials, simulate goalAs
    for cond in ['2d', '3d']:
        trials = load_trials(trials, subj, cond)
        groups = group_trials(trials, coherence_grouper, False)
        for coh, ts in groups.iteritems():
            ts_all = as_x_y(trials)
            # key by theta?

def main(subj, indir, outdir):
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
    INFILE = os.path.join(BASEDIR, 'data.json')
    INDIR = os.path.join(BASEDIR, 'res', indir)
    OUTDIR = os.path.join(BASEDIR, 'res', outdir)
    TRIALS = load_json(INFILE)
    mix(TRIALS, INDIR, OUTDIR, subj)

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--indir", required=True, type=str, help="The directory from which fits will be loaded.")
parser.add_argument('-o', "--outdir", required=True, type=str, help="The directory to which fits will be written.")
parser.add_argument('-s', "--subj", default='SUBJECT', choices=['SUBJECT', 'ALL'] + all_subjs, type=str, help="Specify subject like HUK")
args = parser.parse_args()

if __name__ == '__main__':
    main(args.subj, args.indir, args.outdir)
