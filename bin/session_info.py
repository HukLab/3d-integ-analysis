BINS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30, 0.45, 0.90, 1.30, 2.00]

good_subjects = {
    '2d': ['huk', 'lnk'],
    '3d': ['huk', 'ktz', 'lkc'],
    # '2d': ['huk', 'klb', 'krm', 'lnk'],
    # '3d': ['huk', 'klb', 'krm', 'ktz', 'lkc'],
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
        'ktz': [5],
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
