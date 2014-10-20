#!/bin/bash

OUTDIR=$1
MODEL_SUBJ="krm"
NELBOWS=2
NBINS_THRESH=20
NBOOTS_THRESH=1000
NBOOTS_SAT_EXP=0
RESAMPLE=5

# CREATE OUTDIR
echo "Creating output directory $OUTDIR"
mkdir -p $OUTDIR

# PSYCHOMETRIC FCN -- pcor vs coh
# by dotmode, SUBJ
echo "Psychometric function for $MODEL_SUBJ..."
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --ignore-dur --subj $MODEL_SUBJ
# by dotmode, ALL
echo "Psychometric function for ALL..."
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR --ignore-dur
# by duration, ALL, (2D, 3D)
echo "Psychometric function for ALL (2D, 3D)..."
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR

# SURFACE PLOT
python pd_io.py -l --nbins $NBINS_THRESH --outdir $OUTDIR

# SAT_EXP -- pcor vs dur
# by dotmode, SUBJ
echo "SAT_EXP for $MODEL_SUBJ..."
python pcorVsDurByCoh.py -l --fit --collapse-coh -r $RESAMPLE --outdir $OUTDIR --subj $MODEL_SUBJ
# by dotmode, ALL
echo "SAT_EXP for ALL..."
python pcorVsDurByCoh.py -l --fit --collapse-coh -r $RESAMPLE --outdir $OUTDIR
# by coherence, ALL, (2D, 3D)
echo "SAT_EXP for ALL (2D, 3D)..."
python pcorVsDurByCoh.py -l --fit  --nboots $NBOOTS_SAT_EXP -r $RESAMPLE --outdir $OUTDIR

# echo "Psychometric function for $MODEL_SUBJ (2D, 3D)..."
# python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --subj $MODEL_SUBJ --nboots $NBOOTS_THRESH -r $RESAMPLE --outdir $OUTDIR
