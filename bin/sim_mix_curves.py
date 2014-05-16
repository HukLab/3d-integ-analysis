import numpy as np
import matplotlib.pyplot as plt

from tools import color_list
import saturating_exponential
import huk_tau_e

satexp_rand = lambda x, (a, b, t): np.random.binomial(1, saturating_exponential.saturating_exp(x,a,b,t))
rand_sign = lambda: np.random.randint(2)*2 - 1
first_above = lambda x, As_sorted: next(a for a in As_sorted if a >= x)
first_below = lambda x, As_sorted: next(a for a in reversed(As_sorted) if a <= x)
interpolate_p = lambda pgoal, pbelow, pabove: (pgoal-pbelow)/(pabove-pbelow)

delta = 20.0
As = [0.5 + i/delta for i in xrange(int(delta/2 + 1))]
B = 0.5
T = 500
min_dur = 0.04
max_dur = 2.0

N = 1000*1
nbins = 10
xs = np.linspace(min_dur, max_dur, N)
# xs = np.logspace(np.log10(min_dur), np.log10(max_dur), N)
pcts = 100*np.arange(0.1, 1.0, 1.0/nbins)
quantiles = np.percentile(xs, pcts)
bins = [min_dur] + [q for q in quantiles] + [max_dur]

Agoals = [A + 1/(delta*2.0) for A in As]
N2 = N/5

def simulate_data():
    """
    generate N trials for len(As) curves with saturation points at As
    """
    curves = {}
    for A in sorted(As):
        curves[A] = [(x, satexp_rand(x, (A, B, T))) for x in xs]
    return curves

def interpolate_curves(curves, fit_map, bins, nperbin):
    bins_lr = [(bins[i], bins[i+1]) for i in xrange(len(bins)-1)]
    As_sorted = sorted(fit_map.keys()) # want to use fitted As, not original ones
    goal_curves = {}
    for A in sorted(Agoals):
        try:
            a2, a1 = first_above(A, As_sorted), first_below(A, As_sorted)
        except StopIteration:
            continue
        c1 = curves[fit_map[a1]]
        c2 = curves[fit_map[a2]]
        p = interpolate_p(A, a1, a2) # refers to probability of picking from c2
        data = []
        for (lbin, rbin) in bins_lr:
            c1bin = [(x,y) for x,y in c1 if lbin <= x < rbin] # between lbin, rbin
            c2bin = [(x,y) for x,y in c2 if lbin <= x < rbin] # between lbin, rbin
            c1bininds, nc2bininds = xrange(len(c1bin)), xrange(len(c2bin))
            cur_data = [c1bin[np.random.choice(c1bininds)] if np.random.binomial(1, p) else c2bin[np.random.choice(nc2bininds)] for _ in xrange(nperbin)]
            data.extend(cur_data)
        goal_curves[A] = data
    return goal_curves

def fit(cs, bins, fit=1):
    fs, bs = [], []
    fit_map = {}
    for a in sorted(cs):
        data = cs[a]
        bindata = huk_tau_e.binned_ps(data, bins)
        bindata = zip([i for i in sorted(bindata)], [bindata[i][0] for i in sorted(bindata)])
        bs.append(bindata)

        if fit == 0:
            th = saturating_exponential.fit(data, (None, B, T), quick=True, guesses=[(a,)])
            if not th:
                Af = 0.5
            else:
                Af = th[0]['x'][0]
        elif fit == 1:
            th, Af = huk_tau_e.huk_tau_e(data, B, bins)
            Tf = th[0]['x'][0]
            print (T, Tf)
        else:
            Af = np.mean(zip(*bindata[-2:])[1])

        print (a, Af)
        fit_map[Af] = a
        df = [(x, saturating_exponential.saturating_exp(x, Af, B, T)) for x in xs]
        fs.append(df)
    return fs, bs, fit_map

def plot(fitdata, bindata, clrs, lin='solid'):
    for f, b, clr in zip(fitdata, bindata, clrs):
        xsf, ysf = zip(*f)
        plt.plot(xsf, ysf, color=clr, linestyle=lin)
        xsb, ysb = zip(*b)
        plt.scatter(xsb, ysb, color=clr, linestyle=lin)

def main():
    curves = simulate_data()
    f1, b1, fit_map = fit(curves, bins)
    print '---'
    goal_curves = interpolate_curves(curves, fit_map, bins, N2)
    f2, b2, _ = fit(goal_curves, bins)

    plt.clf()
    plot(f1, b1, color_list(len(f1)))
    plot(f2, b2, color_list(len(f1)), 'dashed')
    plt.title('')
    plt.xlabel('time (ms)')
    plt.ylabel('% correct')
    plt.xlim([min(xs), max(xs)])
    plt.ylim([0.4, 1.0])
    plt.show()

if __name__ == '__main__':
    main()
