QUICK_FIT = True
NBOOTS = 0
NBOOTS_BINNED_PS = 1000 # also used by huk fit
FIT_IS_PER_COH = {'huk': False, 'sat-exp': False, 'drift': True, 'twin-limb': False, 'quick_1974': True}
DEFAULT_THETA = {
    'huk': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'sat-exp': {'A': 1.0, 'B': 0.5, 'T': 0.001},
    'drift': {'K': 0.0, 'X0': 0.038},
    'twin-limb': {'X0': 0.0, 'S0': 1.0, 'P': 0.0},
    'quick_1974': {'A': 0.0, 'B': 0.0},
}

LINESTYLE_MAP = {'huk': 'dotted', 'sat-exp': 'solid', 'drift': 'dashed', 'twin-limb': 'solid', 'quick_1974': 'dashdot'}
COLOR_MAP = {'2d': 'g', '3d': 'r'}
MARKER_MAP = {'2d': 's', '3d': 's'}

BINS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30, 0.45, 0.90, 1.30, 2.00]
# BINS = [0.04, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00]

all_subjs = ['huk', 'klb', 'krm', 'lnk', 'lkc']
good_subjects = {
    # '2d': ['huk', 'lnk'],
    # '3d': ['huk', 'lnk', 'lkc'],
    '2d': ['huk', 'klb', 'krm', 'lnk'],
    '3d': ['huk', 'klb', 'krm', 'lnk', 'lkc'],
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
        'klb': range(106, 117),#range(1, 106),
        'krm': [],
        'lnk': [], #[5],
        'lkc': [],
    }
}

good_cohs = {
    '2d': [0.03, 0.06, 0.12, 0.25, 0.5, 1],
    '3d': [0.03, 0.06, 0.12, 0.25, 0.5, 1],
    # '2d': [0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1],
    # '3d': [0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1],
}
bad_cohs = {
    '2d': [0, 0.01],
    '3d': [0, 0.01, 0.1, 0.15, 0.22, 0.3, 0.4, 0.6],
    # '2d': [],
    # '3d': [0.1, 0.15, 0.22, 0.3, 0.4, 0.6],
}
assert not any([good_cohs[cond] == bad_cohs[cond] for cond in good_cohs])
