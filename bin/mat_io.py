import csv
import glob
import os.path
import argparse

import pandas as pd
import scipy.io as sio

from trial import Trial, Session
from settings import all_subjs
from pd_io import SESSIONS_COLS, TRIALS_COLS

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

def parse_session_index(infile, keyword):
    """
    example filename: runDots_KTZ_-_huk_czuba2dABC_000_(1)_wkspc (w/ KEYWORD=='ABC')
    """
    fname = os.path.splitext(os.path.split(infile)[-1])[0]
    tmp = fname.split('_-_')
    error_str = lambda fname, exp: 'WARNING ({1}): Ignoring file {0}.mat'.format(fname, exp)
    if tmp[0] != 'runDots_KTZ':
        # print error_str(fname, 1)
        return
    tmp = tmp[1].split('_')
    if len(tmp) != 5:
        # print error_str(fname, 2)
        return
    subj, dottype, nums, sess_no, wkspc = tmp
    if len(dottype) != 7 + len(keyword) or not dottype.startswith('czuba'): # 'czuba2d'
        # print error_str(fname, 3)
        return
    if not dottype.endswith(keyword):
        # print error_str(fname, 7)
        return
    if len(nums) != 3: # '000'
        # print error_str(fname, 4)
        return
    if wkspc != 'wkspc':
        # print error_str(fname, 5)
        return
    out = sess_no.split('(')[1].split(')')[0]
    if not out.isdigit(): # '(1)'
        # print error_str(fname, 6)
        return
    return int(out)

def load_mat(infile, keyword, outdir='/Volumes/LKCLAB/Users/Leor/2012-TemporalIntegration/runDots_KTZ_data/longDur'):
    mat = sio.loadmat(infile)
    session_index = parse_session_index(infile, keyword)
    if session_index is None and outdir and 'longDur' in infile:
        fname = os.path.split(infile)[-1]
        outfile = os.path.abspath(os.path.join(outdir, fname))
        print 'RENAMING: {0}'.format(outfile)
        os.rename(infile, outfile)
    if session_index is None:
        return []
    session = load_session(mat, session_index)
    return load_trials(mat, session)

def load(datadir, keyword):
    infiles = glob.glob(os.path.join(datadir, '*.mat'))
    trials = []
    for infile in infiles:
        trials.extend(load_mat(infile, keyword))
    master_trial_sort = lambda t: (t.session.subject, t.session.dotmode, t.session.index, t.coherence, t.duration)
    return sorted(trials, key=master_trial_sort)

def compare_sessions_csv(si1, si2):
    rs2 = [list(x) for x in pd.read_csv(si2, index_col='index').values]
    rs3 = list(rs2)
    if si1:
        rs1 = [list(x) for x in pd.read_csv(si1, index_col='index').values]
        ok_unfound = 0
        sad_unfound = 0
        for r1 in rs1:
            try:
                rs3.pop(rs3.index(r1))
            except ValueError:
                pass
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
        if sad_unfound:
            print 'ERROR: Missing {0} important rows!'.format(sad_unfound)
    print 'Found {0} new rows:'.format(len(rs2))
    for r3 in rs3:
        print r3

def trials_to_csv(trials, sessions_file, trials_file):
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
        csvwriter.writerow(SESSIONS_COLS)
        for i, s in enumerate(ss):
            csvwriter.writerow(s_str(i+1, s))

def main(indir, outdir, outfile1, outfile2, keyword, old_sessions_file):
    CSV_SESSIONS_FILE = os.path.join(outdir, outfile1 + ('' if outfile1.endswith('.csv') else '.csv'))
    CSV_TRIALS_FILE = os.path.join(outdir, outfile2 + ('' if outfile2.endswith('.csv') else '.csv'))

    trials = load(indir, keyword)
    trials_to_csv(trials, CSV_SESSIONS_FILE, CSV_TRIALS_FILE)
    compare_sessions_csv(old_sessions_file, CSV_SESSIONS_FILE)

if __name__ == '__main__':
    """
    e.g. for normalDur, longDur:
        * python mat_io.py -f sessions-2.csv
                           -g trials-2.csv
                           -j ../data/sessions.csv
        * python mat_io.py -i /Volumes/LKCLAB/Users/Leor/2012-TemporalIntegration/runDots_KTZ_data/longDur
                           -f sessions-longDur-2.csv
                           -g trials-longDur-2.csv
                           -j ../data/sessions-longDur.csv
                           -k longDur
    """
    DATADIR = '/Volumes/LKCLAB/Users/Leor/2012-TemporalIntegration/runDots_KTZ_data/normalDur'
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    BASEDIR = os.path.abspath(os.path.join(CURDIR, '..', 'data'))
    OLD_SESSIONS_FILE = os.path.join(BASEDIR, 'sessions.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', default=DATADIR, type=str)
    parser.add_argument('-o', '--outdir', default=BASEDIR, type=str)
    parser.add_argument('-f', '--sessions_outfile', required=True, type=str)
    parser.add_argument('-g', '--trials_outfile', required=True, type=str)
    parser.add_argument('-k', '--keyword', default='', type=str)
    parser.add_argument('-j', '--old_sessions_file', default=None, type=str)
    args = parser.parse_args()
    main(args.indir, args.outdir, args.sessions_outfile, args.trials_outfile, args.keyword, args.old_sessions_file)
