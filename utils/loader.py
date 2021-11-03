import os.path as osp
from collections import namedtuple

import numpy as np
import pandas as pd
from user_config import DATA_DIR


LINE = namedtuple('Line', ['NAME', 'NST', 'STAT', 'Run', 'Dis'])  # Line info


def get_victoria_data(line_name='Victoria', read=True):
    filepath = osp.join(DATA_DIR, f'{line_name}.npy')

    if read and osp.exists(filepath):
        data = np.load(filepath, allow_pickle=True).item()
    else:
        # Read data
        dft = pd.read_csv('./data/mine/tube-info.csv')  # Line Info
        dfod = pd.read_csv('./data/mine/OD15.csv')  # Time Variant demand
        dfrun = pd.read_csv('./data/mine/tube_runtime.csv')
        dfl = pd.read_excel('./data/source/Misc/OD-station-line-time.xls')  # station-inflow

        rename = pd.read_csv('./data/mine/tube_status.csv')
        rename = rename[rename.line == line_name][['sname', 'snaptan']]
        rename = rename.set_index('sname').to_dict()['snaptan']

        for col_ in dfl.columns[:3]:
            dfl[col_] = dfl[col_].str.strip()

        col = dfod.columns[10:-8]

        dfl = dfl[dfl.LINE == line_name]  # inner line passenger demand
        dft = dft[dft.LINE == line_name]
        dfrun = dfrun[dfrun.line == line_name.lower()]

        dfs = dft[['STAT', 'SNLC']].drop_duplicates(keep='first')
        stat, snlc = zip(*dfs.values)
        stat = list(stat)
        snpt = [rename[station][-3:] for station in stat]

        # real demand
        dmd = np.zeros((len(stat), 2, len(col)))
        for idx, row in dfl.iterrows():
            dmd[stat.index(row[0]), int(row[2] == 'N'), :] = row[10:-8].values.reshape((1, -1))

        # inner line demand
        dfod = dfod[dfod.ONLC.isin(snlc) & dfod.DNLC.isin(snlc)]
        tvd = np.zeros(shape=(len(snlc), len(snlc), len(col)))
        for i in range(len(snlc)):
            onlc = snlc[i]
            for j in range(len(snlc)):
                dnlc = snlc[j]
                od = dfod[(dfod.ONLC == onlc) & (dfod.DNLC == dnlc)][col].values
                if od.shape[0] == 0:
                    od = np.zeros(shape=[1, len(col)])
                tvd[i, j] = od

        for i in range(len(stat)):
            outbound = dmd[i, 1] / tvd[i, np.arange(len(stat)) > i].sum(0)
            inbound = dmd[i, 0] / tvd[i, np.arange(len(stat)) < i].sum(0)
            outbound[np.isinf(outbound) | np.isnan(outbound) | (outbound == 0)] = 1
            inbound[np.isinf(inbound) | np.isnan(inbound) | (outbound == 0)] = 1
            for j in range(len(stat)):
                tvd[i, j] *= outbound if j > i else inbound

        tvd = np.repeat(tvd / 15, repeats=15, axis=2)
        dfrun = dfrun.sort_values('direction', kind='mergesort', ascending=False)
        dfrun.sort_index(inplace=True)
        run = (dfrun.loc[:, 'unimpeded'].to_numpy() * 60).astype(int)
        dis = dfrun.loc[:, 'distance'].to_numpy()

        data = dict(name=line_name,
                    length=len(dfs),
                    routes=list(snpt),
                    run=run,
                    distance=(dis * 1000).astype(np.float64),
                    demand=tvd)
        np.save(f'./data/{line_name}.npy', data)

    return data
