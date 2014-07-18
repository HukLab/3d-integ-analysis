# PSYCHOMETRIC FCN
# -----------------

# need weibull params, threshold as well
pcor vs coh by dotmode, SUBJ
pcor vs coh by dotmode, ALL

# need weibull params, thresholds as well
pcor vs coh by duration, ALL, 2D
pcor vs coh by duration, ALL, 3D

# SAT_EXP
# --------
# need sat-exp params as well

# pcor vs dur by dotmode, SUBJ
python pcorVsDurByCoh.py --subj SUBJ --join-dotmode
# pcor vs dur by dotmode, ALL
python pcorVsDurByCoh.py --join-dotmode

# need sat-exp params as well
pcor vs dur by coherence, ALL, 2D
pcor vs dur by coherence, ALL, 3D

# MISC
# ----------

# need twin-limb fits as well
coh-thresh vs dur by dotmode, ALL

tau vs coh by dotmode, ALL
