NBOOTS = 0
NBOOTS_BINNED_PS = 1000 # also used by huk fit
QUICK_FIT = True
# FITS = {'huk': False, 'sat-exp': False, 'drift': True}
FITS = {'huk': True, 'sat-exp': True, 'drift': True}

DEFAULT_THETA = {'A': 1.0, 'B': 0.5, 'T': 0.001, 'K': 0}

BINS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30, 0.45, 0.90, 1.30, 2.00]

all_subjs = ['huk', 'klb', 'krm', 'lnk', 'lkc']
good_subjects = {
    '2d': ['huk', 'lnk'],
    '3d': ['huk', 'lnk', 'lkc'],
    # '2d': ['huk', 'klb', 'krm', 'lnk'],
    # '3d': ['huk', 'klb', 'krm', 'lnk', 'lkc'],
}

bad_sessions = {
    '2d': {
        'huk': [8],
        'klb': [1, 12, 15, 16, 17, 18, 19, 20],
        'krm': [],
        'lnk': [],
    },
    '3d': {
        'huk': [],
        'klb': [],
        'krm': [],
        'lnk': [5],
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
