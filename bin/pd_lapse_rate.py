import argparse
import numpy as np
import matplotlib.pyplot as plt

from pd_io import load, default_filter_df

MIN_COH = 1.0

def all(min_coh=MIN_COH):
    df = default_filter_df(load())
    df2 = df[df.coherence >= 0.5]
    df3 = df2.groupby(['subj', 'dotmode', 'coherence', 'number'], as_index=False)['correct'].agg(np.mean)
    return df3

def main(subj, dotmode, min_coh=MIN_COH):
    df = default_filter_df(load())
    df2 = df[(df.subj == subj) & (df.dotmode == dotmode) & (df.coherence >= MIN_COH)]
    df3 = df2.groupby(['coherence', 'number'], as_index=False)['correct'].agg(np.mean)

    df3.groupby('coherence').plot('number','correct', linestyle='-', marker='o')
    plt.title('Lapse rate for {0} {1}'.format(subj, dotmode))
    plt.xlabel('session #')
    plt.ylim([None, 1.0])
    plt.show()
    return df3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=True, type=str, help="")
    parser.add_argument("--dotmode", required=True, type=str, help="")
    args = parser.parse_args()
    main(args.subj, args.dotmode)
