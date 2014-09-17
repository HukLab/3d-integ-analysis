import pd_io
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisqprob
from matplotlib import pyplot as plt

def load(min_dur=0.3, max_dur=1.2, min_coh=0.03, max_coh=0.5):
    fltrs = [pd_io.make_gt_filter('real_duration', min_dur), pd_io.make_lt_filter('real_duration', max_dur)]
    fltrs.extend([pd_io.make_gt_filter('coherence', min_coh), pd_io.make_lt_filter('coherence', max_coh)])
    return pd_io.load(None, fltrs, 'both')

def test(df, formula):
    return smf.glm(formula, df, family=sm.families.Binomial()).fit()
    # res = smf.ols(formula, df).fit()
    # print res.summary()

def sample_wr(df, n):
    return df.ix[np.random.choice(df.index.values, n)]

def lrt(df, f0, f1, n=5000, N=100):
    """
    draw n samples with replacement from each isLongDur and is NormalDur
    run likelihood ratio test comparing labeled model and unlabeled model
    repeat N times
    """
    rs = []
    llfs = []
    dfA = df[df['isLongDur'] == 1]
    dfB = df[df['isLongDur'] == 0]
    for _ in xrange(N):
        dfcA = sample_wr(dfA, n)
        dfcB = sample_wr(dfB, n)
        dfc = dfcA.append(dfcB)
        t = test(dfc, f1)
        t0 = test(dfc, f0)
        ll = t.llf
        ll0 = t0.llf
        # assert ll > ll0
        llfs.append((ll, ll0))
        rs.append(chisqprob(-2*np.log(ll0/ll), 3)) # df=3: two interactions, one extra constant
    return rs, llfs

def plot(df, res):
    df['mu'] = res.mu
    colors = ['k', 'r']
    for i, (grp, dfc) in enumerate(df.groupby('isLongDur')):
        dfc = dfc.sort('coherence')
        plt.scatter(dfc['coherence'], dfc['mu'], s=3, color=colors[i])
        plt.scatter(dfc['coherence'].unique(), dfc.groupby('coherence').mean()['correct'], s=25, color=colors[i])
    plt.show()

def main(method=1, takeLog=False):

    """
    isLongDur does interaction term for method==1. here's how:
        - use likelihood ratio test between two models:
            * one with two duration and two coherence weights
            * one with one duration and one coherence weight
    """
    df = load()
    df['constant'] = 1
    df['correct'] = df['correct'].astype('float')
    df['isLongDur'] = df['isLongDur'].astype('float')

    if takeLog:
        df['duration'] = np.log(1000*df['duration'])
        df['coherence'] = np.log(100*df['coherence'])

    if method == 0:
        formula = 'correct ~ is2D + isLongDur + duration + coherence'
        df['is2D'] = (df['dotmode'] == '2d')
        df['is2D'] = df['is2D'].astype('float')
        res = test(df, formula)
        plot(df, res)
    elif method == 1:
        formula0 = 'correct ~  duration + coherence'
        df['isNormalDur'] = 1 - df['isLongDur']
        formula = 'correct ~ isLongDur*duration + isLongDur*coherence + isNormalDur*duration + isNormalDur*coherence'
        for dotmode, df_dotmode in df.groupby('dotmode'):
            print '-------'
            print dotmode
            print '-------'
            rs, ls = lrt(df, formula0, formula)
            # plt.scatter(*zip(*ls))
            # plt.show()
            print 'Likelihood ratio test (isLongDur): mean p-value={0}'.format(np.mean(rs))
    else:
        df['isNormalDur'] = 1 - df['isLongDur']
        formula = 'correct ~ isLongDur*duration + isLongDur*coherence + isNormalDur*duration + isNormalDur*coherence'
        for dotmode, df_dotmode in df.groupby('dotmode'):
            res = test(df_dotmode, formula)
            print '-------'
            print dotmode
            print '-------'
            test = 'isLongDur = 0, isNormalDur = 0'
            print test, res.wald_test(test)

if __name__ == '__main__':
    main()
