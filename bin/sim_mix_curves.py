import numpy as np
import matplotlib.pyplot as plt

import saturating_exponential
import huk_tau_e

satexp = lambda x, (a, b, t): a - (a-b)*np.exp(-x/t)
satexp_rand = lambda x, (a, b, t): np.random.binomial(1, satexp(x, (a,b,t)))
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
xs = np.logspace(np.log10(min_dur), np.log10(max_dur), N)*1000
quantiles = np.percentile(xs, 100*np.arange(0, 1 + 1.0/nbins, 1.0/nbins))
bins = [(quantiles[i], quantiles[i+1]) for i in xrange(len(quantiles)-1)]

Agoals = [A + 1/(delta*2.0) for A in As]
N2 = N

def simulate_data():
    """
    generate N trials for len(As) curves with saturation points at As
    """
    curves = {}
    for A in As:
        curves[A] = [(x, satexp_rand(x, (A, B, T))) for x in xs]
    return curves

def interpolate_curves(curves, bins, nperbin):
    As_sorted = sorted(curves.keys())
    goal_curves = {}
    for A in Agoals:
        try:
            a2, a1 = first_above(A, As_sorted), first_below(A, As_sorted)
        except StopIteration:
            continue
        c1 = curves[a1]
        c2 = curves[a2]
        p = interpolate_p(A, a1, a2) # refers to probability of picking from c2
        print A, (a1, a2, p)
        data = []
        for (lbin, rbin) in bins:
            c1bin = [(x,y) for x,y in c1 if lbin <= x < rbin] # between lbin, rbin
            c2bin = [(x,y) for x,y in c2 if lbin <= x < rbin] # between lbin, rbin
            c1bininds, nc2bininds = xrange(len(c1bin)), xrange(len(c2bin))
            cur_data = [c1bin[np.random.choice(c1bininds)] if np.random.binomial(1, p) else c2bin[np.random.choice(nc2bininds)] for _ in xrange(nperbin)]
            data.extend(cur_data)
        goal_curves[A] = data
    return goal_curves

def ps_bin_data(data, bins):
    ps = []
    for (lbin, rbin) in bins:
        databin = [y for x,y in data if lbin <= x < rbin] # between lbin, rbin
        ps.append((lbin, np.mean(databin)))
    return ps

def fit(cs, bins):
    fs, bs = [], []
    for a, data in cs.iteritems():
        bindata = ps_bin_data(data, bins)
        bs.append(bindata)

        # th = saturating_exponential.fit(data, (None, B, None), quick=True)[0]
        # Tf = th['x'][1]
        # th = saturating_exponential.fit(data, (None, B, T), quick=False, guesses=[(a,)])[0]
        # Af = th['x'][0]

        # bns = [l for l,r in bins]
        # th, Af = huk_tau_e.huk_tau_e(data, B, bns)
        # Tf = th[0]['x'][0]
        # Af = np.mean(zip(*bindata[-2:])[1])
        # print (Af, np.mean(zip(*bindata[-2:])[1]))
        
        print (a, Af)
        df = [(x, satexp(x, (Af, B, T))) for x in xs]
        fs.append(df)
    return fs, bs

def plot(fitdata, bindata, clr):
    for f, b in zip(fitdata, bindata):
        xsf, ysf = zip(*f)
        plt.plot(xsf, ysf, color=clr)
        xsb, ysb = zip(*b)
        plt.scatter(xsb, ysb, color=clr)

def main():
    curves = simulate_data()
    goal_curves = interpolate_curves(curves, bins, N2)
    print '---'
    f1, b1 = fit(curves, bins)
    print '---'
    f2, b2 = fit(goal_curves, bins)

    plt.clf()
    plot(f1, b1, 'b')
    plot(f2, b2, 'r')
    plt.title('')
    # plt.xscale('log')
    plt.xlabel('time (ms)')
    plt.ylabel('% correct')
    plt.xlim([min(xs), max(xs)])
    plt.ylim([0.4, 1.0])
    plt.show()

if __name__ == '__main__':
    main()
