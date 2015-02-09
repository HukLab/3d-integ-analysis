#!/bin/bash

OUTDIR=$1
NELBOWS=2
NBINS_THRESH=20
NBOOTS_THRESH=1000
RESAMPLE=5

# CREATE OUTDIR
echo "Creating output directory $OUTDIR"
mkdir -p $OUTDIR

echo "Psychometric function for ALL (2D, 3D)..."

python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR

MODEL_SUBJ="krm"
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ

MODEL_SUBJ="klb"
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ

MODEL_SUBJ="huk"
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ

MODEL_SUBJ="lkc"
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ

MODEL_SUBJ="lnk"
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ
