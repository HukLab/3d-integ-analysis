import argparse
import numpy as np
import matplotlib.pyplot as plt

from tools import color_list
from pd_io import load

def plot(df):
    durinds = df['duration_index'].unique()
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    cols = color_list(len(durinds))
    colmap = dict((di, col) for di, col in zip(durinds, cols))

    fig = plt.figure()
    ax = plt.subplot(111)
    for di in durinds:
        dfc = df[df['duration_index'] == di]
        xsp, ysp = zip(*dfc.groupby('coherence').agg(np.mean)['correct'].reset_index().values)
        ax.plot(xsp, ysp, color=colmap[di], label="%0.2f" % durmap[di], marker='o', linestyle='-')
    plt.title('% correct vs. coherence, by duration')
    plt.xlabel('coherence')
    plt.ylabel('% correct')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.4, 1.05])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def main(args):
    df = load(args)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        plot(df_dotmode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--dotmode", required=False, type=str, help="")
    args = parser.parse_args()
    main(vars(args))
