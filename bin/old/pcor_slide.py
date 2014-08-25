import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from scipy.optimize import minimize

from pd_io import load

DEFAULT_RES = 50

def diff(f2d, f3d, (i, j, k), res):
    inrange = lambda a, b: 0 <= a <= res-1 and 0 <= b <= res-1
    return np.nansum([pow(f2d[x,y] - f3d[x+i, y+j] + k, 2) for x in xrange(res) for y in xrange(res) if inrange(x+i, y+j)])

def slide(f2d, f3d, res):
    minx = (0, 0, 0)
    minv = None
    for i in np.arange(-res/2 + 1, res/2):
        if i % 5 == 0:
            print minx, minv
        for j in np.arange(-res/2 + 1, res/2):
            for k in [0]:
                v = diff(f2d, f3d, (i, j, k), res)
                if v < minv or minv is None:
                    minx = (i, j, k)
                    minv = v
    return minx, minv

def frame(xs, ys, zs, (xi, yi), durmap):
    xs = np.log(100.*np.array(xs))
    ys = np.array([1000.*durmap[y] for y in ys])
    return griddata(xs, ys, zs, xi, yi, interp='linear')

def ranges(df, resX, resY, durmap):
    xs = np.log(100.*np.array(df['coherence']))
    min_xs, max_xs = xs.min(), xs.max()
    xi = np.linspace(min_xs, max_xs, resX)

    # min_ys, max_ys = df['duration_index'].min(), df['duration_index'].max()
    # yi = np.linspace(min_ys, max_ys, resY)

    min_ys, max_ys = durmap[df['duration_index'].min()], durmap[df['duration_index'].max()]
    yi = np.logspace(np.log10(1000.*min_ys), np.log10(1000.*max_ys), resY)

    return xi, yi

def plot(f2d, f3d, xi, yi, minx):
    nx, ny = f2d.shape
    xinds2d = max(-minx[0], 0), min(nx, nx - minx[0])
    yinds2d = max(-minx[1], 0), min(ny, ny - minx[1])
    xinds3d = max(minx[0], 0), min(nx, nx + minx[0])
    yinds3d = max(minx[1], 0), min(ny, ny + minx[1])
    
    x = xi[xinds2d[0]:xinds2d[1]]
    y = yi[yinds2d[0]:yinds2d[1]]
    X, Y = np.meshgrid(x, y)
    Z2d = f2d[yinds2d[0]:yinds2d[1], xinds2d[0]:xinds2d[1]]
    Z3d = f3d[yinds3d[0]:yinds3d[1], xinds3d[0]:xinds3d[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X0, Y0 = np.meshgrid(xi, yi)
    ax.plot_wireframe(X0, np.log(Y0), f2d, color='g')

    ax.plot_wireframe(X, np.log(Y), Z3d, color='r')
    ax.set_xlabel('log(coherence)')
    ax.set_ylabel('log(duration (ms))')
    ax.set_zlabel('p correct')
    # ax.set_xlim([1.0, 5.0])
    # ax.set_ylim([3.5, 7.0])
    # ax.set_zlim([0.3, 1.0])
    plt.show()

def summary(xi, yi, minx, minv):
    print
    print 'COHERENCE'
    for x1, x2 in [(xi[i], xi[i + minx[0]]) for i in xrange(len(xi)) if 0 <= i + minx[0] < len(xi)]:
        print '%0.2f' % np.exp(x1), '%0.2f' % np.exp(x2)
    print
    print 'DURATION'
    for y1, y2 in [(yi[i], yi[i + minx[1]]) for i in xrange(len(yi)) if 0 <= i + minx[1] < len(yi)]:
        print '%0.2f' % y1, '%0.2f' % y2
    print
    print minx
    print minv

def main(args, res):
    """
    RES = 50:
    --------------
    sbj coh dur
    --------------
    huk (-3, 3, 0)
    lnk (18, 9, 0)
    klb (5,  8, 0)
    krm (7, 14, 0)
    ALL (7, 10, 0)
    --------------
    """
    df = load(args)
    df = df[df['coherence'] > 0.0]
    data = {}
    durmap = dict(df.groupby('duration_index')['duration'].agg(min).reset_index().values)
    xi, yi = ranges(df, res, res, durmap)

    for dotmode, df_dotmode in df.groupby('dotmode'):
        dfc = df_dotmode.groupby(['coherence', 'duration_index'], as_index=False)['correct'].aggregate(np.mean)
        xs, ys, zs = zip(*dfc.values)
        data[dotmode] = frame(xs, ys, zs, (xi, yi), durmap)
    assert '2d' in data and '3d' in data

    minx, minv = slide(data['2d'], data['3d'], res)
    summary(xi, yi, minx, minv)
    plot(data['2d'], data['3d'], xi, yi, minx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=False, type=str, help="")
    parser.add_argument("--res", required=False, type=int, default=DEFAULT_RES, help="")
    args = parser.parse_args()
    main({'subj': args.subj}, args.res)
