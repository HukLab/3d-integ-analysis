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
