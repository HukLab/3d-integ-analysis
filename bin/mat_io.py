import csv
import json
import glob
import os.path

import scipy.io as sio

from dio import MyEncoder
from trial import Trial, Session
from session_info import all_subjs

SUBJS = set()
def load_session(mat, session_index):
    ps = mat['dv']['params'][0,0]
    ps = dict(zip(ps.dtype.names, ps[0,0]))
    if ps['dottype'][0] != 'czuba':
        return None
    subj_map = lambda x: x if x != 'ktz' else 'lnk' # KTZ -> LNK
    subj = subj_map(ps['subj'][0].lower())
    if subj not in all_subjs:
        return None
    dotmode = ps['dotmode'][0]
    binEdges = list(ps['binEdges'][0])
    return Session(subj, dotmode, binEdges, session_index)

def load_trials(mat, session):
    if session is None:
        return []
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
    return load_trials(mat, session)

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

def compare_sessions_csv(si1, si2):
    import pandas as pd
    rs1 = [list(x) for x in pd.read_csv(si1, index_col='index').values]
    rs2 = [list(x) for x in pd.read_csv(si2, index_col='index').values]
    ok_unfound = 0
    sad_unfound = 0
    for r1 in rs1:
        try:
            i2 = rs2.index(r1)
            d2v = rs2[:i2] + rs2[i2+1:]
        except ValueError:
            if r1[0] not in all_subjs:
                ok_unfound += 1
            else:
                sad_unfound += 1
                print 'ERROR: Not found: {0}'.format(r1)
    print 'Found {0} old rows.'.format(len(rs1) - ok_unfound - sad_unfound)
    print 'Dropped {0} rows with bad subjects.'.format(ok_unfound)
    print 'Found {0} new rows.'.format(len(rs2))
    if sad_unfound:
        print 'ERROR: Missing {0} important rows!'.format(sad_unfound)

def trials_to_csv(trials, sessions_file, trials_file):
    SESS_COLS = [u'index', u'subj', u'dotmode', u'number']
    TRIALS_COLS = [u'index', u'session_index', u'trial_index', u'coherence', u'duration', u'duration_index', u'direction', u'response', u'correct']

    ss = []
    t_str = lambda i, s_i, t: [i, s_i, t.index, t.coherence, t.duration, t.duration_index, t.direction, t.response, t.correct]
    with open(trials_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(TRIALS_COLS)
        for i, t in enumerate(trials):
            if t.session not in ss:
                ss.append(t.session)
            s_i = ss.index(t.session)
            csvwriter.writerow(t_str(i+1, s_i+1, t))

    s_str = lambda i, s: [i, s.subject, s.dotmode, s.index]
    with open(sessions_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(SESS_COLS)
        for i, s in enumerate(ss):
            csvwriter.writerow(s_str(i+1, s))

def main():
    DATADIR = '/Volumes/LKCLAB/Users/Leor/2012-TemporalIntegration/runDots_KTZ_data'
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))

    OUTFILE = os.path.join(BASEDIR, 'data-2.json')
    trials = mat_to_json(DATADIR, OUTFILE)

    CSV_SESSIONS_FILE = os.path.join(BASEDIR, 'csv', 'sessions-2.csv')
    CSV_TRIALS_FILE = os.path.join(BASEDIR, 'csv', 'trials-2.csv')
    trials_to_csv(trials, CSV_SESSIONS_FILE, CSV_TRIALS_FILE)

    compare_sessions_csv(CSV_SESSIONS_FILE.replace('-2', ''), CSV_SESSIONS_FILE)

if __name__ == '__main__':
    main()
