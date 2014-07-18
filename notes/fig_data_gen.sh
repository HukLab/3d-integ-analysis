#!/bin/bash

OUTDIR=data-$(date +%Y-%m-%d)
MODEL_SUBJ="KRM"
NBOOTS_THRESH=0
NBOOTS_SAT_EXP=0

# PSYCHOMETRIC FCN
# -----------------

# NOT CURRENTLY WRITING DATA: need binned pcors, weibull params, threshold
# pcor vs coh by dotmode, SUBJ
python pcorVsCohByDur.py --subj $MODEL_SUBJ --join-dotmode --thresh --outdir $OUTDIR --nboots $NBOOTS_THRESH
# pcor vs coh by dotmode, ALL
python pcorVsCohByDur.py --join-dotmode --thresh --outdir $OUTDIR --nboots $NBOOTS_THRESH

# NOT CURRENTLY WRITING DATA: need binned pcors, weibull params, threshold, twin-limb fits
# pcor vs coh by duration, ALL, (2D, 3D)
python pcorVsCohByDur.py --thresh --outdir $OUTDIR --plot-thresh --nboots $NBOOTS_THRESH

# SAT_EXP
# --------

# NOT CURRENTLY WRITING DATA: need binned pcors, sat-exp params
# pcor vs dur by dotmode, SUBJ
python pcorVsDurByCoh.py --subj $MODEL_SUBJ --join-dotmode --outdir $OUTDIR
# pcor vs dur by dotmode, ALL
python pcorVsDurByCoh.py --join-dotmode --outdir $OUTDIR

# pcor vs dur by coherence, ALL, (2D, 3D)
python fitCurveVsDurByCoh.py --is_agg_subj --fits sat-exp --nboots $NBOOTS_SAT_EXP --outdir $OUTDIR
