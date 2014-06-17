import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

def make_surface(xs, ys, zs, resX=50, resY=50):
    xi = np.linspace(min(xs), max(xs), resX)
    yi = np.linspace(min(ys), max(ys), resY)
    Z = griddata(xs, ys, zs, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def plot_inner(data, fig, color, durmap):
    if len(data) == 2:
        ax = fig.gca(projection='2d')
        ax.scatter(*data, color=color)
    elif len(data) == 3:
        ax = fig.gca(projection='3d')
        x,y,z = data
        x = np.log(100.*np.array(x))
        y = np.log([1000.*durmap[i] for i in y])
        ax.scatter(x,y,z, color=color)
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
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    df2 = df_dotmode[['coherence', 'duration_index', 'correct']]
    df3 = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)

def plot(df, show=True, outfile=None, fig=None):
    if len(df) == 0:
        return
    if fig is None:
        fig = plt.figure()
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        df2 = df_dotmode[['coherence', 'duration_index', 'correct']]
        df3 = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)
        plot_inner(zip(*df3.values), fig, 'g' if dotmode == '2d' else 'r', durmap)
    save_or_show(show, outfile)
