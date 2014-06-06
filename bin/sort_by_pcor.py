import argparse

import numpy as np
import matplotlib.pyplot as plt

from pd_io import load
from tools import color_list

colmap = lambda items: dict((coh, col) for coh, col in zip([0]*1 + items, color_list(len(items) + 1, "YlGnBu")))

def main(args):
    df = load(args)
    cols = ['coherence','duration_index']
    dfc = df.groupby(cols + ['dotmode'], as_index=False)['correct'].agg(np.mean).sort(cols).reset_index()
    # dfc = df.groupby(list(reversed(cols)) + ['dotmode'], as_index=False)['correct'].agg(np.mean).sort(list(reversed(cols))).reset_index()
    for dm, dc in dfc.groupby('dotmode', as_index=False):
        dc.plot('index', 'correct', linestyle='-', marker='o')
    plt.show()
    return

    cm = colmap(dfc['duration_index'].unique())
    for (dm, di), dc in dfc.groupby(['dotmode', 'duration_index']):
        if dm == '3d':
            dc['coherence_offset'] = np.exp(np.log(dc['coherence']) + 0.1)
        else:
            dc['coherence_offset'] = dc['coherence']
        dc.plot('coherence_offset', 'correct', color=cm[di], linestyle='', marker='o')

    # dfc2d = dfc[dfc['dotmode']=='2d']
    # dfc2d.plot('coherence', 'correct', color='g', linestyle='', marker='o')

    # dfc3d = dfc[dfc['dotmode']=='3d']
    # dfc3d['coherence_offset'] = np.exp(np.log(dfc3d['coherence']) + 0.1)
    # dfc3d.plot('coherence_offset', 'correct', color='r', linestyle='', marker='o')

    plt.xlim([0.02, 1.4])
    plt.ylim([0.3, 1.05])
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--subj", required=False)
    args = parser.parse_args()
    main(vars(args))
