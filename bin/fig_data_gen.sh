#!/bin/bash

OUTDIR=$1
MODEL_SUBJ="krm"
NELBOWS=2
NBINS_THRESH=20
NBOOTS_THRESH=1000
NBOOTS_SAT_EXP=0

# CREATE OUTDIR
# ------------
echo "Creating output directory $OUTDIR"
mkdir -p $OUTDIR

# PSYCHOMETRIC FCN -- pcor vs coh
# -----------------
# by dotmode, SUBJ
echo "Psychometric function for $MODEL_SUBJ..."
python pmf_fit.py -l -e $NELBOWS --nboots $NBINS_THRESH --subj $MODEL_SUBJ --ignore-dur --nboots $NBOOTS_THRESH --outdir $OUTDIR
# by dotmode, ALL
echo "Psychometric function for ALL..."
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --ignore-dur --nboots $NBOOTS_THRESH --outdir $OUTDIR
# by duration, ALL, (2D, 3D)
echo "Psychometric function for ALL (2D, 3D)..."
python pmf_fit.py -l -e $NELBOWS --nbins $NBINS_THRESH --nboots $NBOOTS_THRESH --outdir $OUTDIR

# # SAT_EXP -- pcor vs dur
# # --------
# # by dotmode, SUBJ
# echo "SAT_EXP for $MODEL_SUBJ..."
# python pcorVsDurByCoh.py -l --subj $MODEL_SUBJ --fit --outdir $OUTDIR
# # by dotmode, ALL
# echo "SAT_EXP for ALL..."
# python pcorVsDurByCoh.py -l --fit --outdir $OUTDIR
# # by coherence, ALL, (2D, 3D)
# echo "SAT_EXP for ALL (2D, 3D)..."
# python fitCurveVsDurByCoh.py -l --is_agg_subj --fits sat-exp --nboots $NBOOTS_SAT_EXP --outdir $OUTDIR
