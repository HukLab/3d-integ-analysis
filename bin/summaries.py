import pandas as pd
from pd_io import load, load_df, SESSIONS_INFILE_2, TRIALS_INFILE_2, resample_by_grp

def main():
    def summary(df, key=['subj', 'dotmode', 'isLongDur']):
        rows = []  
        for k, dg in df.groupby(key, as_index=False):
            tm = dg['session_index']
            nts = []
            for k1, dh in dg.groupby('session_index'):
                nts.append(len(set(dh['trial_index'])))
            v = '{0} - {1}'.format(min(nts), max(nts)) if min(nts) != max(nts) else str(min(nts))
            row = list(k) + [len(set(tm.values)), len(tm), v]
            rows.append(row)
        return pd.DataFrame(rows, columns=key + ['# sessions', '# trials', '# trials/session'])

    # data actually collected
    df1 = load_df()
    df1['isLongDur'] = False
    dfA = summary(df1)
    df2 = load_df(SESSIONS_INFILE_2, TRIALS_INFILE_2)
    df2['isLongDur'] = True
    dft = summary(df2)
    dfA = dfA.append(dft)

    # data used in analysis
    df3 = load({}, None, 'both')
    dfB = summary(df3)

    # combine
    key = ['isLongDur', 'subj', 'dotmode']
    dfA = dfA.sort(key).reset_index()
    dfB = dfB.sort(key).reset_index()
    dfA['# trials analyzed'] = dfB['# trials']
    dfA['# trials/session analyzed'] = dfB['# trials/session']
    keys = ["# sessions", "# trials", "# trials analyzed", "# trials/session", "# trials/session analyzed"]
    print dfA.set_index(key)[keys].to_csv()
    print dfA.groupby('isLongDur')[keys].sum()
    print dfA.groupby('dotmode')[keys].sum()

    print resample_by_grp(df3)

if __name__ == '__main__':
    main()