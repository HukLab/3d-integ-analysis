import os.path
import argparse
import subprocess

BINDIR = 'bin'
SUBJS = ['lnk', 'huk', 'klb', 'krm']
NBOOTS = 20

def main(basedir, fit_sat_exp=False, subjs=SUBJS):
    if fit_sat_exp:
        BASEDIR = os.path.join(basedir, 'vs-dur-fit')
        subprocess.call(['python', os.path.join(BINDIR, 'fitCurveVsDurByCoh.py'), '--fits', 'sat-exp', '--SUBJ', 'ALL', '--outdir', OUTDIR], shell=True)
        subprocess.call(['python', os.path.join(BINDIR, 'fitCurveVsDurByCoh.py'), '--fits', 'sat-exp', '--SUBJ', 'SUBJECT', '--outdir', OUTDIR], shell=True)
        subprocess.call(['python', os.path.join(BINDIR, 'plotCurveVsDurByCoh.py'), '--fits', 'sat-exp', '--SUBJ', 'ALL', '--indir', OUTDIR, '--outdir', OUTDIR], shell=True)
        subprocess.call(['python', os.path.join(BINDIR, 'plotCurveVsDurByCoh.py'), '--fits', 'sat-exp', '--SUBJ', 'SUBJECT', '--indir', OUTDIR, '--outdir', OUTDIR], shell=True)

    # python pd_pcorVsDurByCoh.py --savefig --outdir [OUTDIR]
    # python pd_pcorVsDurByCoh.py --savefig  --outdir [OUTDIR] --subj [SUBJ]
    BASEDIR = os.path.join(basedir, 'vs-dur-raw')
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsDurByCoh.py'), '--savefig', '--outdir', OUTDIR], shell=True)
    for SUBJ in SUBJS:
        subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsDurByCoh.py'), '--savefig', '--outdir', OUTDIR, '--subj', SUBJ], shell=True)

    # pd_pcorVsCohByDur.py --savefig --outdir [OUTDIR]
    # python pd_pcorVsCohByDur.py --savefig  --outdir [OUTDIR] --subj [SUBJ]
    BASEDIR = os.path.join(basedir, 'vs-coh-raw')
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--savefig', '--outdir', OUTDIR], shell=True)
    for SUBJ in SUBJS:
        subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--savefig', '--outdir', OUTDIR, '--subj', SUBJ], shell=True)

    # python pd_pcorVsCohByDur.py --thresh --nboots [NBOOTS] --plot-thresh --savefig --outdir [OUTDIR]
    # python pd_pcorVsCohByDur.py --thresh --subj [SUBJ] --nboots [NBOOTS] --plot-thresh --savefig --outdir [OUTDIR]
    BASEDIR = os.path.join(basedir, 'vs-coh-fit-boots')
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--thresh', '--nboots', NBOOTS, '--plot-thresh', '--savefig', '--outdir', OUTDIR], shell=True)
    for SUBJ in SUBJS:
        subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--thresh', '--subj', SUBJ, '--nboots', NBOOTS, '--plot-thresh', '--savefig', '--outdir', OUTDIR], shell=True)

    # python pd_pcorVsCohByDur.py --thresh --nboots 0 --plot-thresh --savefig --outdir [OUTDIR]
    # python pd_pcorVsCohByDur.py --thresh --subj [SUBJ] --nboots 0 --plot-thresh --savefig --outdir [OUTDIR]
    BASEDIR = os.path.join(basedir, 'vs-coh-fit')
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--thresh', '--nboots', 0, '--plot-thresh', '--savefig', '--outdir', OUTDIR], shell=True)
    for SUBJ in SUBJS:
        subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsCohByDur.py'), '--thresh', '--subj', SUBJ, '--nboots', 0, '--plot-thresh', '--savefig', '--outdir', OUTDIR], shell=True)

    BASEDIR = os.path.join(basedir, 'vs-pcor')
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsSig.py'), '--savefig', '--outdir', OUTDIR], shell=True)
    subprocess.call(['python', os.path.join(BINDIR, 'pd_pcorVsSig.py'), '--unfold', '--savefig', '--outdir', OUTDIR], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, default='.')
    parser.add_argument('--fit-sat-exp', required=False, action='store_true', default=False)
    args = parser.parse_args()
    main(args.outdir, args.fit_sat_exp)
