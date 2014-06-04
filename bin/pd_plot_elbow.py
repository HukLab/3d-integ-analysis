import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scatter_plots import plot_by_key, plot_info

CURDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.abspath(os.path.join(CURDIR, '..'))
inf1 = os.path.join(BASEDIR, 'csv', 'elbow-lines.csv')
inf2 = os.path.join(BASEDIR, 'csv', 'elbow-splits.csv')

df1 = pd.read_csv(inf1, index_col='subj')
df2 = pd.read_csv(inf2, index_col='subj')
assert all(df1.keys() == [u'dotmode', u'line', u'slope', u'intercept'])
assert all(df2.keys() == [u'dotmode', u'split'])

def main():
    fig = plt.figure()
    key = 'slope'
    ymin, ymax = plot_by_key(df1[df1['line'] == 1], key, label='pre-elbow')
    ymin, ymax = plot_by_key(df1[df1['line'] == 2], key, label='post-elbow', color='white')
    plot_info(key + ' (75% coherence threshold / duration)', (ymin, ymax), True)

    fig = plt.figure()
    key = 'intercept'
    ymin, ymax = plot_by_key(df1[df1['line'] == 1], key, label='pre-elbow')
    ymin, ymax = plot_by_key(df1[df1['line'] == 2], key, label='post-elbow', color='white')
    plot_info(key + ' (75% coherence threshold)', (ymin, ymax), True)

    fig = plt.figure()
    key = 'split'
    ymin, ymax = plot_by_key(df2, key)
    plot_info('elbow duration (ms)', (ymin, ymax))

    plt.show()

if __name__ == '__main__':
    main()
