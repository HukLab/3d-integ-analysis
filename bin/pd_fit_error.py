import numpy as np
import matplotlib.pyplot as plt

from pd_io import load, interpret_filters, filter_df, default_filter_df, plot
from pd_plot import plot_inner
from plotCurveVsDurByCoh import load_pickle
from drift_diffuse import drift_diffusion
from saturating_exponential import saturating_exp

def plot_fit_error(df, indir, subj, cond, fit):

    fig = plt.figure()

    # observed data
    df2 = df[['coherence', 'duration_index', 'correct']]
    df3 = df2.groupby(['coherence', 'duration_index'], as_index=False).aggregate(np.mean)
    # plot_inner(zip(*df3.values), fig, 'c')

    # fit data
    indir = os.path.join(BASEDIR, 'res', indir)
    res = load_pickle(indir, subj, cond)['fits'][fit]
    if len(res) == 1:
        fnd = True
        if fit == 'drift':
            X0, K = res[0]['X0'], res[0]['K']
        elif fit == 'sat-exp':
            A, B, T = res[0]['A'], res[0]['B'], res[0]['T']
    else:
        fnd = False
        res = dict((float(r), d) for r,d in res.iteritems())
    durs = dict(np.fliplr(df[['duration', 'duration_index']].groupby('duration_index', as_index=False).aggregate(np.min).values))
    cohs = df.coherence.unique()
    vals = []
    for c in sorted(cohs):
        for t in sorted(durs):
            if not fnd:
                if fit == 'drift':
                    X0, K = res[c][0]['X0'], res[c][0]['K']
                elif fit == 'sat-exp':
                    A, B, T = res[c][0]['A'], res[c][0]['B'], res[c][0]['T']
            if fit == 'drift':
                val = drift_diffusion((c, t), K, X0)
            elif fit == 'sat-exp':
                val = saturating_exp(t, A, B, T)
            val2 = float(df3[(df3.coherence==c) & (df3.duration_index==durs[t])].correct)
            # vals.append((c, durs[t], val))
            vals.append((c, durs[t], val2-val))
    plot_inner(zip(*vals), fig, 'b')

    plt.show()

def main(args):
    df = load()
    df = filter_df(df, interpret_filters(args))
    df = default_filter_df(df)
    plot(df)

    # plot_fit_error(df, 'sat-exp', 'ALL', args['dotmode'], 'sat-exp')
    # plot_fit_error(df, 'drift-fits-free', 'ALL', args['dotmode'], 'drift')
    # plot_fit_error(df, 'drift-fits-og', 'ALL', args['dotmode'], 'drift')

if __name__ == '__main__':
    main()
