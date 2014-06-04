import argparse
import numpy as np
import matplotlib.pyplot as plt

from tools import color_list
from pd_io import load

def plot_inner(df, ax):
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cohs = sorted(df['coherence'].unique())
    colmap = dict((coh, col) for coh, col in zip([0]*2 + cohs, color_list(len(cohs) + 2, "YlGnBu")))
    for coh, df_coh in df.groupby('coherence'):
        isp, ysp = zip(*df_coh.groupby('duration_index').agg(np.mean)['correct'].reset_index().values)
        xsp = [durmap[i]*1000 for i in isp]
        ax.plot(xsp, ysp, color=colmap[coh], label="%0.2f" % coh, marker='o', linestyle='-')

def plot(df, dotmode):
    fig = plt.figure()
    ax = plt.subplot(111)
    plot_inner(df, ax)
    plt.title('{0}: % correct vs. duration, by coherence'.format(dotmode))
    plt.xlabel('duration')
    plt.ylabel('% correct')
    plt.xscale('log')
    plt.xlim([30, 1050])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def main(args):
    df = load(args)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        plot(df_dotmode, dotmode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    args = parser.parse_args()
    main(vars(args))
