import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import logistic

from tools import color_list

F = lambda x, (a,b,l): 0.5 + (1-0.5-l)*logistic.cdf(x, loc=a, scale=b)
Finv = lambda thresh_val, (a,b,l): logistic.ppf((thresh_val - 0.5)/(1-0.5-l), loc=a, scale=b)

make_durmap = lambda df: dict(list(df.groupby('duration_index')['real_duration'].agg(min).reset_index().values) + [(0, 0)])
make_colmap = lambda durinds: dict((di, col) for di, col in zip(durinds, color_list(len(durinds))))
is_number = lambda x: x and not (np.isnan(x) | np.isinf(x))
label_fcn = lambda di: "%0.2f" % di
color_fcn = lambda dotmode: 'g' if dotmode == '2d' else 'r'

def plot_info_pmf():
    plt.title('% correct vs. coherence')
    plt.xlabel('coherence')
    plt.ylabel('% correct')
    # plt.xlim([0.0, 1.05])
    plt.xscale('log')
    plt.ylim([0.4, 1.05])

def plot_info_threshes():
    plt.title('sensitivity vs. duration')
    plt.xlabel('duration')
    plt.ylabel('sensitivity')
    # plt.xlim([0.0, 1.05])
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim([0.4, 1.05])

def plot_pmf_with_data((xs, ys), theta, thresh, color, label):
    plt.scatter(xs, ys, color=color, label=label, marker='o')
    xsc = np.linspace(min(xs), 1.0)#max(xs))
    plt.axvline(thresh, color=color, linestyle='--')
    plt.text(0.1, 0.5, ("thresh=%0.2f, lapse=%0.2f" % (thresh, theta[-1])))
    # if is_number(thresh):
    #     plt.text(thresh + 0.01, 0.5, 'threshold={0}%'.format(int(thresh*100)), color=color)
    ysf = F(xsc, theta)
    plt.plot(xsc, ysf, color=color, linestyle='-')

def plot_logistics(df_pts, df_fts):
    colmap = make_colmap(sorted(df_pts['di'].unique()))
    vals = {}
    for dotmode, dfd in df_pts.groupby('dotmode'):
        vals[dotmode] = []
        for di, dfp in dfd.groupby('di'):
            color = colmap[di]
            dff = df_fts[(df_fts['dotmode'] == dotmode) & (df_fts['di'] == di)]
            label = label_fcn(dff['dur'].values[0])
            theta = dff[['loc', 'scale', 'lapse']].dropna().mean().values
            thresh = dff['thresh'].dropna().mean()
            vals[dotmode].append((di, dfp[['x', 'y']].corr()['x']['y']))
            # theta = [dff[key].values[0] for key in ['loc', 'scale', 'lapse']]
            plot_pmf_with_data((dfp['x'], dfp['y']), theta, thresh, color, label)
        plot_info_pmf()
        plt.show()
    for dotmode, vls in vals.iteritems():
        plt.scatter(*zip(*vls), color='g' if dotmode == '2d' else 'r')
        plt.title('correlation between coherence and percent correct')
        plt.xlabel('duration index')
        plt.ylabel('r')
    plt.show()

def plot_line(xs, (m0, b0), color, show_text):
    f = lambda x: (x**m0)*np.exp(b0)
    xsf = np.linspace(min(xs), max(xs))
    plt.plot(xsf, f(xsf), color=color)
    if show_text:
        plt.text(np.mean(xs), np.mean(f(xsf)), 'm0={0:.2f}'.format(m0), color=color)

def plot_one_elbow(xs, (x0, m0, b0, m1, b1), color, show_text):
    x0 = np.exp(x0)
    f1 = lambda x: (x**m0)*np.exp(b0)
    f2 = lambda x: (x**m1)*np.exp(b1)
    xs0 = np.linspace(min(xs), x0)
    xs1 = np.linspace(x0, max(xs))
    plt.plot(xs0, f1(xs0), color=color)
    plt.plot(xs1, f2(xs1), color=color)
    plt.axvline(x0, color=color, linestyle='--')
    if show_text:
        plt.text(x0, min(f1(xs0)) + min(f1(xs0))/2, 'x0={0:.0f}'.format(x0), color=color)
    if not list(xs0) or not list(xs1): return
    if show_text:
        plt.text(np.mean(xs0), max(f1(xs0)), 'm={0:.2f}'.format(m0), color=color)
        plt.text(np.mean(xs1), max(f2(xs0)), 'm={0:.2f}'.format(m1), color=color)

def plot_two_elbows(xs, (x0, m0, b0, m1, b1, x1, m2, b2), color, show_text):
    x0 = np.exp(x0)
    x1 = np.exp(x1)
    f1 = lambda x: (x**m0)*np.exp(b0)
    f2 = lambda x: (x**m1)*np.exp(b1)
    f3 = lambda x: (x**m2)*np.exp(b2)
    xs0 = np.linspace(min(xs), x0)
    xs1 = np.linspace(x0, x1)
    xs2 = np.linspace(x1, max(xs))
    plt.plot(xs0, f1(xs0), color=color)
    plt.plot(xs1, f2(xs1), color=color)
    plt.plot(xs2, f3(xs2), color=color)
    if show_text:
        plt.axvline(x0, color=color, linestyle='--')
        plt.axvline(x1, color=color, linestyle='--')
        plt.text(x0, min(f1(xs0)) + min(f1(xs0))/2, 'x0={0:.0f}'.format(x0), color=color)
        plt.text(x1, min(f2(xs1)) + min(f2(xs1))/2, 'x1={0:.0f}'.format(x1), color=color)
        if not list(xs0) or not list(xs1): return
        plt.text(np.mean(xs0), max(f1(xs0)), 'm0={0:.2f}'.format(m0), color=color)
        plt.text(np.mean(xs1), max(f2(xs1)), 'm1={0:.2f}'.format(m1), color=color)
        plt.text(np.mean(xs2), max(f3(xs2)), 'm2={0:.2f}'.format(m2), color=color)

def plot_elbow(xs, df, color, show_text):
    get_th = lambda keys: (df[key].values[0] for key in keys)
    if 'm2' in df:
        th = get_th(['x0', 'm0', 'b0', 'm1', 'b1', 'x1', 'm2', 'b2'])
        plot_two_elbows(xs, th, color, show_text)
    elif 'm1' in df:
        th = get_th(['x0', 'm0', 'b0', 'm1', 'b1'])
        plot_one_elbow(xs, th, color, show_text)
    else:
        th = get_th(['m0', 'b0'])
        plot_line(xs, th, color, show_text)

def plot_threshes(df_pts, df_elbs):
    """
    n.b. thresh and loc are basically the same; scale is similar; also a weird dependence on lapse

    beautiful relation in 3d between scale and thresh, but not at all for 2d!
        - 3d has huge thresh and scale values...
    """
    YKEY = 'sens' # 'thresh'
    for dotmode, dfp in df_pts.groupby('dotmode'):
        color = color_fcn(dotmode)
        dfc = dfp.groupby('dur')[YKEY].agg([np.mean, lambda vs: np.std(vs, ddof=1)]).reset_index()
        xs, ys, yerrs = zip(*dfc.values)
        plt.scatter(xs, ys, marker='o', s=35, label=dotmode, color=color)
        plt.errorbar(xs, ys, yerr=yerrs, fmt=None, ecolor=color)
        show_text = True
        if not df_elbs.empty:
            dfea = df_elbs[df_elbs['dotmode'] == dotmode]
            for bi, dfe in dfea.groupby('bi'):
                xsa = np.linspace(min(xs), max(xs))
                plot_elbow(xsa, dfe, color, show_text)
                show_text = False
    plot_info_threshes()
    plt.show()
