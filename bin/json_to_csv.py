import csv
import json
import glob
import os.path
import argparse

# elbow
E0 = ['subj', 'dotmode', 'x', 'y']
E1 = ['subj', 'dotmode', 'x0', 'm0', 'b0', 'm1', 'b1']
def parse_elbow(obj, subj, dotmode):
    """
    objs[dotmode] = {'binned': (xs, ys)}
    objs[dotmode]['binned'] = (xs, ys)
    objs[dotmode]['fit'] = (x0, m0, b0, m1, b1)
    """
    a, b = [], []

    xsb, ysb = obj['binned']
    for x, y, in zip(xsb, ysb):
        r1 = {'subj': subj, 'dotmode': dotmode, 'x': x, 'y': y}
        a.append(r1)

    r2 = {'subj': subj, 'dotmode': dotmode}
    r2.update(dict((k,v) for k,v in zip(['x0', 'm0', 'b0', 'm1', 'b1'], obj['fit'])))
    b.append(r2)
    return a, b

# thresh -- n.b. (di, dur) will be empty if '_by_dotmode'
T0 = ['subj', 'dotmode', 'di', 'dur', 'x', 'y', 'ntrials']
T1 = ['subj', 'dotmode', 'di', 'dur', 'bi', 'thresh', 'loc', 'scale', 'lapse']
def parse_thresh(obj, subj, dotmode, is_by_dotmode):
    """
    objs = [{'di': di, 'dur': durmap[di], 'obj': val}, ...]
    val = {'binned': (xs, ys, zs), 'fit': [(theta, thresh), ...]}
    """
    a, b = [], []
    obj_to_data = lambda ob: [ob['binned'][0], ob['binned'][1], ob['binned'][2], ob['fit']]

    def make_rows(di, dur, val):
        xsb, ysb, zsb, fits = obj_to_data(val)
        r1 = [{'subj': subj, 'dotmode': dotmode, 'di': di, 'dur': dur, 'x': x, 'y': y, 'ntrials': z} for x, y, z in zip(xsb, ysb, zsb)]
        r2 = [{'subj': subj, 'dotmode': dotmode, 'di': di, 'dur': dur, 'bi': bi, 'thresh': thresh, 'loc': theta[0], 'scale': theta[1], 'lapse': theta[2]} for bi, (theta, thresh) in enumerate(fits)]
        a.extend(r1)
        b.extend(r2)

    for vals in obj:
        di = vals['di']
        dur = vals['dur']
        make_rows(vals['di'] if is_by_dotmode else '', vals['dur'] if is_by_dotmode else '', vals['obj'])
    return a, b

# sat-exp-dotmode
SA1 = ['subj', 'dotmode', 'is_bin_or_fit', 'x', 'y']
SA2 = ['subj', 'dotmode', 'A', 'B', 'T']
def parse_sat_exp_dotmode(obj, subj, dotmode):
    """
    obj = {'binned': {'xs': list(xsp), 'ys': list(ysp)}, 'fit': {'xs': list(xs), 'ys': list(ys), 'theta': list(th)}}
    """
    a, b = [], []
    xsb, ysb = obj['binned']['xs'], obj['binned']['ys']
    xsf, ysf, th = [obj['fit'][key] for key in ['xs', 'ys', 'theta']]
    assert len(th) == 3
    for x, y in zip(xsb, ysb):
        r1 = {'subj': subj, 'dotmode': dotmode, 'is_bin_or_fit': 'bin', 'x': x, 'y': y}
        a.append(r1)
    for x, y in zip(xsf, ysf):
        r1 = {'subj': subj, 'dotmode': dotmode, 'is_bin_or_fit': 'fit', 'x': x, 'y': y}
        a.append(r1)
    r2 = {'subj': subj, 'dotmode': dotmode, 'A': th[0], 'B': th[1], 'T': th[2]}
    b.append(r2)
    return a, b

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
    if 'elbow' in infile:
        return parse_elbow(obj, subj, dotmode)
    elif 'thresh_by_dotmode' in infile:
        return parse_thresh(obj, subj, dotmode, False)
    elif 'thresh' in infile:
        return parse_thresh(obj, subj, dotmode, True)
    elif 'pcorVsDur' in infile:
        return parse_sat_exp_dotmode(obj, subj, dotmode)
    elif 'fitCurve' in infile:
        return parse_sat_exp(obj, subj, dotmode)
    else:
        raise Exception("ERROR interpreting internal filetype.")

def write(rss, outfile_strf):
    for i, rs in enumerate(rss):
        with open(outfile_strf.format('params' if i == 1 else 'pts'), 'w') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=rs[0].keys())
            csvwriter.writeheader()
            csvwriter.writerows(rs)

def main(indir, outdir):
    for infile in glob.glob(os.path.join(indir, '*.json')):
        infile = os.path.abspath(os.path.join(indir, infile))
        outfile_strf = os.path.abspath(os.path.join(outdir, os.path.splitext(os.path.split(infile)[-1])[0] + '-{0}.csv'))
        with open(infile) as f:
            obj = json.load(f)
            rss = parse(infile, obj)
            write(rss, outfile_strf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, type=str)
    parser.add_argument('--outdir', required=True, type=str)
    args = parser.parse_args()
    main(args.indir, args.outdir)
