import csv
import json
import glob
import os.path
import argparse

# sat-exp
SB1 = ['subj', 'dotmode', 'coh', 'di', 'dur', 'pc', 'se', 'ntrials']
SB2 = ['subj', 'dotmode', 'bi', 'coh', 'A', 'B', 'T']
def parse_sat_exp(obj, subj, dotmode):
    """
    {
    "dotmode": "2d", 
    "fits": {
        "sat-exp": {
            "0.25": [
                {
                    "A": 0.96708753430594385, 
                    "B": 0.5, 
                    "T": 58.908567142286138
                }
            ], 
            "0.5": [
        "binned_pcor": {
            "0.25": {
                "0.17883387208205614": [
                    0.96094263456090634, 
                    0.0049988472426868271, 
                    1412
                ], 
                "0.1222090833748956": [
                    0.91063609684519442, 
    """
    a, b = [], []
    for coh, vals in obj['fits']['binned_pcor'].iteritems():
        coh = float(coh)
        for di, (dur, xs) in enumerate(vals.iteritems()):
            pc, se, n = xs
            r1 = {'subj': subj, 'dotmode': dotmode, 'coh': coh, 'di': di, 'dur': dur, 'pc': pc, 'se': se, 'ntrials': n}
            a.append(r1)
    for coh, items in obj['fits']['sat-exp'].iteritems():
        coh = float(coh)
        for bi, th in enumerate(items):
            r2 = {'subj': subj, 'dotmode': dotmode, 'bi': bi, 'coh': coh, 'A': th['A'], 'B': th['B'], 'T': th['T']}
            b.append(r2)
    return a, b

def parse(infile, obj):
    dotmode = '2d' if '2d' in infile else '3d'
    subj = infile[infile.index(dotmode)-4:infile.index(dotmode)-1]
    if 'fitCurve' in infile:
        return parse_sat_exp(obj, subj, dotmode)
    else:
        raise Exception("ERROR interpreting internal filetype.")

def unique_fname(filename):
    if not os.path.exists(filename):
        return filename
    i = 1
    update_ofcn = lambda infile, i: infile.replace('.json', '-{0}.json'.format(i))
    while os.path.exists(update_ofcn(filename, i)):
        i += 1
    return update_ofcn(filename, i)

def write(rows, outfile_strf):
    for i, rs in enumerate(rows):
        outfile = unique_fname(outfile_strf.format('params' if i == 1 else 'pts'))
        print outfile
        with open(outfile, 'w') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=rs[0].keys())
            csvwriter.writeheader()
            csvwriter.writerows(rs)

def main(indir, outdir):
    for infile in glob.glob(os.path.join(os.path.abspath(indir), '*.json')):
        outfile_strf = os.path.abspath(os.path.join(outdir, os.path.splitext(os.path.split(infile)[-1])[0] + '-{0}.csv'))
        with open(infile) as f:
            obj = json.load(f)
            rows = parse(infile, obj)
            write(rows, outfile_strf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', required=True, type=str)
    parser.add_argument('-o', '--outdir', required=True, type=str)
    args = parser.parse_args()
    main(args.indir, args.outdir)
