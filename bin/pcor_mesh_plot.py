import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

def make_surface(xs, ys, zs, resX=50, resY=50):
    xi = np.linspace(min(xs), max(xs), resX)
    yi = np.linspace(min(ys), max(ys), resY)
    Z = griddata(xs, ys, zs, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def plot_inner(data, fig, color):
    if len(data) == 2:
        ax = fig.gca(projection='2d')
        ax.scatter(*data, color=color)
    elif len(data) == 3:
        ax = fig.gca(projection='3d')
        x,y,z = data
        x = np.log(np.array(x)*100)
        y = np.log(np.array(y)*1000)
        ax.scatter(x, y, z, color=color)
        ax.plot_wireframe(*make_surface(x,y,z), color=color)
        ax.set_xlabel('log(coherence)')
        ax.set_ylabel('log(duration (ms))')
        ax.set_zlabel('p correct')
        ax.set_xlim([0.5, 5.0])
        ax.set_zlim([0.3, 1.0])
    else:
        raise Exception("too many dimensions in d: {0}".format(len(data)))

def save_or_show(show, outfile):
    if not show:
        return
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

def plot_diff(df):
    durmap = dict(df.groupby('duration_index')['real_duration'].agg(min).reset_index().values)
    df2 = df_dotmode[['coherence', 'duration_index', 'correct']]
    df3 = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)

def plot(df, show=True, outfile=None, fig=None):
    if len(df) == 0:
        return
    if fig is None:
        fig = plt.figure()
    durmap = dict(df.groupby('duration_index')['real_duration'].agg(min).reset_index().values)
    df0 = pd.DataFrame()
    for dotmode, df_dotmode in df.groupby('dotmode'):
        df2 = df_dotmode[['coherence', 'duration_index', 'correct']]
        dfc = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)
        dfc['duration_index'] = [durmap[i] for i in dfc['duration_index']]
        plot_inner(zip(*dfc.values), fig, 'g' if dotmode == '2d' else 'r')
        dfc['dotmode'] = dotmode
        df0 = df0.append(dfc)
    save_or_show(show, outfile)
    df0.rename(columns={'duration_index': 'dur', 'coherence': 'coh', 'correct': 'pc'}, inplace=True)
    return df0
