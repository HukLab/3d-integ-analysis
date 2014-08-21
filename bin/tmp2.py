import csv
import glob
import os.path

indir = '../plots/pmf-boots-csv'
outfile = '../plots/pmf-boots.csv'
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

with open(outfile, 'wb') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
    csvwriter.writeheader()
    csvwriter.writerows(rows)
