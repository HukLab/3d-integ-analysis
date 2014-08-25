import csv
import glob
import os.path
import subprocess

def run_at_multiple_bins(outdir):
    """
    2d:
        * first slope: x<15, 15<x<17, 17<x
        * noise at end: high 20s, especially 21 and 25-28
    3d:
        * not that affected when x>15
    """
    for i in xrange(10, 30):
        print 'NBINS = {0}'.format(i)
        proc = subprocess.call('python pmf_fit.py -l -e 0 -n {0} -b 0 --outdir {1}'.format(i, outdir), shell=True)

def concatenate_to_csv(indir, outfile):
    rows = []
    fieldnames = ['lapse', 'loc', 'scale', 'thresh', 'dotmode', 'dur', 'di', 'subj', 'bi']
    for infile in glob.glob(os.path.join(os.path.abspath(indir), '*params.csv')):
        n = int(infile.split('-')[-2])
        print n
        with open(infile, 'rb') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',', fieldnames=fieldnames)
            csvreader.next()
            for item in csvreader:
                row = {'i': item['dur'], 'thresh': item['thresh'], 'S': n}
                for x in ['T', 'TND', 'N', 'cohs', 'K', 'A']:
                    row[x] = 0
                rows.append(row)
    write(outfile, rows)

def write(outfile, rows):
    with open(outfile, 'wb') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        csvwriter.writeheader()
        csvwriter.writerows(rows)

if __name__ == '__main__':
    outdir = '../plots/pmf-boots'
    outfile = '../plots/pmf-boots.csv'
    run_at_multiple_bins(outdir)
    concatenate_to_csv(outdir, outfile)
