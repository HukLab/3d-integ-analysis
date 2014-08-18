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
    plt.title('threshold vs. duration')
    plt.xlabel('duration')
    plt.ylabel('threshold')
    # plt.xlim([0.0, 1.05])
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim([0.4, 1.05])

def plot_pmf_with_data(dfc, theta, thresh, color, label):
    xsp, ysp = zip(*dfc.groupby('coherence', as_index=False)['correct'].agg(np.mean).values)
    plt.scatter(xsp, ysp, color=color, label=label, marker='o')
    xsc = np.linspace(min(xsp), max(xsp))
    plt.axvline(thresh, color=color, linestyle='--')
    # if is_number(thresh):
    #     plt.text(thresh + 0.01, 0.5, 'threshold={0}%'.format(int(thresh*100)), color=color)
    ys = F(xsc, theta)
    plt.plot(xsc, ys, color=color, linestyle='-')

def plot_logistics(df, res):
    colmap = make_colmap(sorted(df['duration_index'].unique()))
    durmap = make_durmap(df)
    for dotmode, df_dotmode in df.groupby('dotmode'):
        for di, df_durind in df_dotmode.groupby('duration_index'):
            if dotmode not in res or di not in res[dotmode]:
                continue
            theta, thresh = res[dotmode][di]['fit'][0]
            color = colmap[di]
            label = label_fcn(durmap[di])
            plot_pmf_with_data(df_durind, theta, thresh, color, label)
        plot_info_pmf()
        plt.show()

def plot_elbow((xs, ys), (x0, m0, b0, m1, b1), color):
    x0 = np.exp(x0)
    f1 = lambda x: (x**m0)*np.exp(b0)
    f2 = lambda x: (x**m1)*np.exp(b1)
    xs0 = np.linspace(min(xs), x0)
    xs1 = np.linspace(x0, max(xs))
    plt.plot(xs0, f1(xs0), color=color)
    plt.plot(xs1, f2(xs1), color=color)
    plt.axvline(x0, color=color, linestyle='--')
    plt.text(x0, min(f1(xs0)) + min(f1(xs0))/2, 'x0={0:.0f}'.format(x0), color=color)
    if not list(xs0) or not list(xs1): return
    plt.text(np.mean(xs0), max(f1(xs0)), 'm={0:.2f}'.format(m0), color=color)
    plt.text(np.mean(xs1), max(f2(xs0)), 'm={0:.2f}'.format(m1), color=color)

def plot_two_elbows((xs, ys), (x0, m0, b0, m1, b1, x1, m2, b2), color):
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
    plt.axvline(x0, color=color, linestyle='--')
    plt.axvline(x1, color=color, linestyle='--')
    plt.text(x0, min(f1(xs0)) + min(f1(xs0))/2, 'x0={0:.0f}'.format(x0), color=color)
    plt.text(x1, min(f2(xs1)) + min(f2(xs1))/2, 'x1={0:.0f}'.format(x1), color=color)
    if not list(xs0) or not list(xs1): return
    plt.text(np.mean(xs0), max(f1(xs0)), 'm0={0:.2f}'.format(m0), color=color)
    plt.text(np.mean(xs1), max(f2(xs1)), 'm1={0:.2f}'.format(m1), color=color)
    plt.text(np.mean(xs2), max(f3(xs2)), 'm2={0:.2f}'.format(m2), color=color)

def plot_threshes(df, res, elbs=None):
    durmap = make_durmap(df)
    for dotmode, res1 in res.iteritems():
        pts = []
        for di, res0 in res1.iteritems():
            pts.extend([(durmap[di], thresh) for theta, thresh in res0['fit']])
        if not pts:
            return
        xs, ys = zip(*pts)
        xs = 1000*np.array(xs)

        df = pd.DataFrame({'xs': xs, 'ys': ys})
        df = df.groupby(xs)['ys'].agg([np.mean, lambda vs: np.std(vs, ddof=1)]).reset_index()
        xs, ys, yerrs = zip(*df.values)

        color = color_fcn(dotmode)
        plt.scatter(xs, ys, marker='o', s=35, label=dotmode, color=color)
        plt.errorbar(xs, ys, yerr=yerrs, fmt=None, ecolor=color)
        # plt.xlim([10, 10000])
        # plt.ylim([0.01, 10])
        if dotmode in elbs:
            # plot_elbow((xs, ys), elbs[dotmode]['fit'], color)
            plot_two_elbows((xs, ys), elbs[dotmode]['fit'], color)
    plot_info_threshes()
    plt.show()
