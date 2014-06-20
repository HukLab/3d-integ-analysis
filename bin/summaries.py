import csv
import os.path
from itertools import groupby

import numpy as np

session_grouper = lambda t: (t.session.subject, t.session.dotmode)
session_ind_grouper = lambda t: (t.session.subject, t.session.dotmode, t.session.index)
subj_grouper = lambda t: t.session.subject
dot_grouper = lambda t: t.session.dotmode
coherence_grouper = lambda t: t.coherence
duration_grouper = lambda t: t.duration_index
raw_duration_grouper = lambda t: t.duration
def group_trials(trials, keyfunc, verbose=False):
    trials = sorted(trials, key=keyfunc)
    groups = {}
    for k, g in groupby(trials, keyfunc):
        groups[k] = list(g)
    if verbose:
        for k in groups:
            print k, len(groups[k])
    assert len(trials) == sum([len(x) for x in groups.values()])
    return groups

def as_x_y(trials):
    return np.array([(t.duration, int(t.correct)) for t in trials])

def as_C_x_y(trials):
    return np.array([([t.coherence, t.duration], int(t.correct)) for t in trials])

def by_coherence(trials, (subj, mode), coh):
    groups = group_trials(trials, session_grouper, False)
    trials = groups[(subj, mode)]
    groups = group_trials(trials, coherence_grouper, False)
    trials = groups[coh]
    return trials

# def write_csvs(trials, trials_csv, sessions_csv):
#     gps = group_trials(trials, session_ind_grouper)
#     with open(sessions_csv, 'wb') as csvfile:
#         csvwriter = csv.writer(csvfile, delimiter=',')
#         csvwriter.writerow(['index', 'subj', 'dotmode', 'number'])
#         session_map = dict((s, i+1) for i,s in enumerate(sorted(gps)))
#         csvwriter.writerows(sorted([(session_map[s],) + s for s in gps]))
#     with open(trials_csv, 'wb') as csvfile:
#         csvwriter = csv.writer(csvfile, delimiter=',')
#         csvwriter.writerow(['index', 'session_index', 'trial_index', 'coherence', 'duration', 'duration_index', 'direction', 'response', 'correct'])
#         for i, t in enumerate(sorted(trials)):
#             row = [i+1, session_map[session_ind_grouper(t)], t.index, t.coherence, t.duration, t.duration_index, t.direction, t.response, t.correct]
#             csvwriter.writerow(row)
