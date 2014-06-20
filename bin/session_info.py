NBOOTS = 0
NBOOTS_BINNED_PS = 1000 # also used by huk fit

QUICK_FIT = False
FIT_IS_COHLESS = {
    'huk': False,
    'sat-exp': False,
    'drift': True,# False, -> enable drift_diffusion_2
    'twin-limb': False,
    'quick_1974': True,
}
DEFAULT_THETA = {
    'huk': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'sat-exp': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'drift': {'K': 12.705, 'X0': 0.0389}, # 12.705 9.816
    'twin-limb': {'X0': 0.0, 'S0': 1.0, 'P': 0.0},
    'quick_1974': {'A': 0.0, 'B': 0.0},
}
THETAS_TO_FIT = {
    'sat-exp': {'A': True, 'B': False, 'T': True},
    'drift': {'K': True, 'X0': False},
    'twin-limb': {'X0': True, 'S0': True, 'P': True},
    'quick_1974': {'A': True, 'B': True},
}

LINESTYLE_MAP = {
    'huk': 'solid',
    'sat-exp': 'solid',
    'drift': 'solid',#'dashed',
    'twin-limb': 'solid',
    'quick_1974': 'dashdot'
}
COLOR_MAP = {'2d': 'g', '3d': 'r'}
MARKER_MAP = {'2d': 's', '3d': 's'}

import numpy as np

NDOTS = 40
FRATE = 60
nframes = lambda duration: np.ceil(duration*FRATE)
nsigframes = lambda duration: nframes(duration) - 1 # takes two to see motion
nsigdots = lambda coherence, duration: (NDOTS*coherence)*nsigframes(duration)

# BINS = list(np.array([2, 3, 5, 8, 12, 18, 26, 36, 48, 62, 76, 94, 114])*(1/60.))
min_dur, max_dur, N = 0.039, 1.2, 10
BINS = list(np.logspace(np.log10(min_dur), np.log10(max_dur), N))
# BINS = list(np.linspace(min_dur, max_dur, N))
# BINS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30, 0.45, 0.90, 1.30, 2.00]

all_subjs = ['huk', 'klb', 'krm', 'lnk', 'lkc']
good_subjects = {
    # '2d': ['huk', 'lnk'],
    # '3d': ['huk', 'lnk', 'lkc'],
    '2d': ['huk', 'klb', 'krm', 'lnk'],
    '3d': ['huk', 'klb', 'krm', 'lnk'],#, 'lkc'],
}

bad_sessions = {
    '2d': {
        'huk': [], #[8],
        'klb': [], #[1, 12, 15, 16, 17, 18, 19, 20],
        'krm': [],
        'lnk': [],
    },
    '3d': {
        'huk': [],
        'klb': range(95), #range(42, 147), #range(106, 117),
        'krm': [],
        'lnk': [], #[5],
        'lkc': [],
    }
}

good_cohs = {
    '2d': [0.03, 0.06, 0.12, 0.25, 0.5],#, 1],
    '3d': [0.03, 0.06, 0.12, 0.25, 0.5]#, 1],
    # '2d': [0, 0.03, 0.06, 0.12, 0.25, 0.5, 1],
    # '3d': [0, 0.03, 0.06, 0.12, 0.25, 0.5, 1],
}
bad_cohs = {
    '2d': [0, 0.01],
    '3d': [0, 0.01, 0.1, 0.15, 0.22, 0.3, 0.4, 0.6],
    # '2d': [0.01],
    # '3d': [0.01, 0.1, 0.15, 0.22, 0.3, 0.4, 0.6],
}
# assert not any([good_cohs[cond] == bad_cohs[cond] for cond in good_cohs])
