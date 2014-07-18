import numpy as np

NBOOTS_BINNED_PS = 1000 # also used by huk fit

QUICK_FIT = False
FIT_IS_COHLESS = {
    'huk': False,
    'sat-exp': False,
    'drift': True, # False, -> enable drift_diffusion_2
    'drift-diff': True, # False
    'twin-limb': False,
    'quick_1974': True,
}
DEFAULT_THETA = {
    'huk': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'sat-exp': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'drift': {'K': 12.705, 'X0': 0.0389}, # 12.705 9.816
    'drift-diff': {'K': 12.705, 'L': 2.0, 'X0': 0.0389}, # 12.705 9.816
    'twin-limb': {'X0': 0.0, 'S0': 1.0, 'P': 0.0},
    'quick_1974': {'A': 0.0, 'B': 0.0},
}
THETAS_TO_FIT = {
    'sat-exp': {'A': True, 'B': False, 'T': True},
    'drift': {'K': True, 'X0': False},
    'drift-diff': {'K': True, 'L': True, 'X0': False},
    'twin-limb': {'X0': True, 'S0': True, 'P': True},
    'quick_1974': {'A': True, 'B': True},
}

LINESTYLE_MAP = {
    'huk': 'solid',
    'sat-exp': 'solid',
    'drift': 'solid',
    'drift-diff': 'solid',
    'twin-limb': 'solid',
    'quick_1974': 'dashdot'
}
COLOR_MAP = {'2d': 'g', '3d': 'r'}
MARKER_MAP = {'2d': 's', '3d': 's'}

NDOTS = 40
FRATE = 60
nframes = lambda duration: np.ceil(duration*FRATE)
actualduration = lambda duration: nframes(duration)/FRATE
nsigframes = lambda duration: nframes(duration) - 1 # takes two to see motion
nsigdots = lambda coherence, duration: (NDOTS*coherence)*nsigframes(duration)

min_dur, max_dur, NBINS = 0.04, 1.2, 10
BINS = list(np.logspace(np.log10(min_dur - 0.001), np.log10(max_dur), NBINS))

AGG_SUBJ_NAME = 'ALL'
all_subjs = ['huk', 'klb', 'krm', 'lnk', 'lkc']
good_subjects = {
    '2d': ['huk', 'klb', 'krm', 'lnk', 'lkc'],
    '3d': ['huk', 'klb', 'krm', 'lnk', 'lkc'],
}

bad_sessions = {
    '2d': {
        'huk': [],
        'klb': [],
        'krm': [],
        'lnk': [],
        'lkc': [],
    },
    '3d': {
        'huk': [],
        'klb': range(95),
        'krm': [],
        'lnk': [],
        'lkc': [],
    }
}

good_cohs = {
    '2d': [0.03, 0.06, 0.12, 0.25, 0.5],#, 1],
    '3d': [0.03, 0.06, 0.12, 0.25, 0.5]#, 1],
}
