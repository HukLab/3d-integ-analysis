import json
import glob
import os.path
import argparse
from scipy.io import savemat

def convert(input):
    """
    since scipy.io.savemat currently has problems with nested dicts containing unicode
    """
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def json_to_mat(infile, outfile):
    with open(infile) as f:
        obj = convert(json.load(f))
        savemat(outfile, obj)

def main(indir, outdir):
    for infile in glob.glob(os.path.join(indir, '*.json')):
        infile = os.path.abspath(os.path.join(indir, infile))
        outfile = os.path.abspath(os.path.join(outdir, os.path.splitext(os.path.split(infile)[-1])[0]))
        json_to_mat(infile, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, type=str)
    parser.add_argument('--outdir', required=True, type=str)
    args = parser.parse_args()
    main(args.indir, args.outdir)
