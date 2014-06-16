import glob
import os.path
import json
from json import JSONEncoder

from trial import Trial, Session

makefn = lambda outdir, subj, cond, name, ext: os.path.join(outdir, '{0}-{1}-{2}.{3}'.format(subj, cond, name, ext))

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def from_json(json_object):
    if 'subject' in json_object:
        keys = ['subject', 'dotmode', 'duration_bins', 'index']
        values = [json_object[key] for key in keys]
        return Session(*values)
    elif 'session' in json_object:
        keys = ['session', 'index', 'coherence', 'duration', 'duration_index', 'direction', 'response', 'correct']
        values = [json_object[key] for key in keys]
        return Trial(*values)
    else:
        return json_object

def load_json(infile):
    with open(infile) as f:
        return json.load(f, object_hook=from_json)

# def parse_filename(infile):
#     return infile.replace('.json', '').split('_')

# def load(data):
#     ps = data['dv']['params']
#     if 'session_number' in ps:
#         session_index = ps['session_number']
#     else:
#         session_index = parse_filename(infile)[-1]
#     subj_map = lambda x: x if x != 'ktz' else 'lnk' # KTZ -> LNK
#     session = Session(subj_map(ps['subj']), ps['dotmode'], ps['binEdges'], session_index)

#     trials = []
#     for tr, coh, dirc, durIdx, dur, resp, crct in data['D']:
#         trial = Trial(session, tr, coh, dur, durIdx, dirc, resp, True if crct == 1 else False)
#         trials.append(trial)
#     return trials

# def sort_trials(infile, outfile):
#     trials = load_master_json(infile)
#     print len(trials)
#     master_trial_sort = lambda t: (t.session.subject, t.session.dotmode, t.session.index, t.coherence, t.duration)
#     trials = sorted(trials, key=master_trial_sort)
#     print len(trials)
#     with open(outfile, 'w') as f:
#         json.dump(trials, f, cls=MyEncoder)

# def jsons_to_json(datadir, outfile):
#     infiles = glob.glob(os.path.join(datadir, '*.json'))
#     trials = []
#     for infile in infiles:
#         with open(infile) as f:
#             data = json.load(f)
#             ts = load(data)
#             trials.extend(ts)
#     master_trial_sort = lambda t: (t.session.subject, t.session.dotmode, t.session.index, t.coherence, t.duration)
#     trials = sorted(trials, key=master_trial_sort)
#     with open(outfile, 'w') as f:
#         json.dump(trials, f, cls=MyEncoder)
#     return len(trials)

# def main():
#     curdir = os.path.dirname(os.path.abspath(__file__))
#     basedir = os.path.abspath(os.path.join(curdir, '..'))
#     datadir = os.path.join(basedir, 'json')
#     outfile = os.path.join(basedir, 'data-2.json')

#     ntrials = jsons_to_json(datadir, outfile)
#     xs = load_json(outfile)
#     assert len(xs) == ntrials
#     print 'SUCCESS'

# if __name__ == '__main__':
#     main()
