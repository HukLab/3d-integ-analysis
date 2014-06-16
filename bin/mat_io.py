import csv
import json
import glob
import os.path

import scipy.io as sio

from dio import MyEncoder
from trial import Trial, Session

def load_session(mat, session_index):
    ps = mat['dv']['params'][0,0]
    ps = dict(zip(ps.dtype.names, ps[0,0]))
    assert ps['dottype'][0] == 'czuba'
    subj = ps['subj'][0]
    dotmode = ps['dotmode'][0]
    binEdges = ps['binEdges'][0]
    subj_map = lambda x: x if x != 'ktz' else 'lnk' # KTZ -> LNK
    return Session(subj_map(subj), dotmode, binEdges, session_index)

def load_trials(mat, session):
    trials = []
    for tr, coh, dirc, durIdx, dur, resp, crct in mat['D']:
        trial = Trial(session, tr, coh, dur, durIdx, dirc, resp, True if crct == 1 else False)
        trials.append(trial)
    return trials

def parse_session_index(infile):
    return os.path.split(infile)[-1].split('_(')[1].split(')')[0]

def load(infile):
    mat = sio.loadmat(infile)
    session = load_session(mat, parse_session_index(infile))
    ts = load_trials(mat, session)

def write(trials, outfile):
    with open(outfile, 'w') as f:
        json.dump(trials, f, cls=MyEncoder)

def mat_to_json(datadir, outfile):
    infiles = glob.glob(os.path.join(datadir, '*.mat'))
    trials = []
    for infile in infiles:
        trials.extend(load(infile))
    master_trial_sort = lambda t: (t.session.subject, t.session.dotmode, t.session.index, t.coherence, t.duration)
    trials = sorted(trials, key=master_trial_sort)
    write(trials, outfile)
    return trials

def trials_to_csv(trials, sessions_file, trials_file):
    SESS_COLS = [u'subj', u'dotmode', u'number']
    TRIALS_COLS = [u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct']

    ss = list(set([t.session for t in trials]))
    s_str = lambda s: [s.subject, s.dotmode, s.index]
    with open(sessions_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(SESS_COLS)
        csvwriter.writerows([s_str(s) for s in ss])

    t_str = lambda t: [t.session.index, t.index, t.coherence, t.duration, t.duration_index, t.direction, t.response, t.correct]
    with open(trials_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(TRIALS_COLS)
        csvwriter.writerows([t_str(t) for t in trials])

def main():
    DATADIR = '/Volumes/LKCLAB/Users/Leor/2012-TemporalIntegration/runDots_KTZ_data'
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))

    OUTFILE = os.path.join(BASEDIR, 'data-2.json')
    trials = mat_to_json(DATADIR, OUTFILE)

    CSV_SESSIONS_FILE = os.path.join(BASEDIR, 'csv', 'sessions-2.csv')
    CSV_TRIALS_FILE = os.path.join(BASEDIR, 'csv', 'trials-2.csv')
    trials_to_csv(trials, CSV_SESSIONS_FILE, CSV_TRIALS_FILE)

if __name__ == '__main__':
    main()
