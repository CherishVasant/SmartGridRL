"""Official CityLearn Challenge 2022 KPIs (Nweye et al. 2023, Appendix A) for every finished run, from its eval_trace.csv.

C = sum_h max(0, E_h * price_h) / same for no battery      (district net load E_h, clipped at district level)
G = sum_h max(0, E_h * carbon_h) / same for no battery
D = mean(R, 1-L): ramping sum|E_h - E_{h-1}| and (1 - monthly load factor, 730-h months), each / no battery
Phase I score  = mean(C, G)          (public leaderboard on the 5 training buildings = our dataset)
Phase III score = mean(C, G, D)
Usage (from the phase 3 folder): python tools/challenge_kpis.py  -> prints a ranking, writes results/challenge_kpis.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / 'results'


def raw(trace):
    e, price, ci = trace['load'].values, trace['pricing'].values, trace['carbon_intensity'].values
    months = [e[i:i + 730] for i in range(0, len(e), 730) if len(e[i:i + 730]) >= 24]
    return {'c': np.maximum(0, e * price).sum(), 'g': np.maximum(0, e * ci).sum(), 'r': np.abs(np.diff(e)).sum(),
            'l': np.mean([1 - m.mean() / m.max() for m in months])}


def challenge_kpis(trace, nb):
    k = raw(trace)
    C, G, R, L = k['c'] / nb['c'], k['g'] / nb['g'], k['r'] / nb['r'], k['l'] / nb['l']
    D = (R + L) / 2
    return {'C': C, 'G': G, 'R': R, '1-L': L, 'D': D, 'phase1': (C + G) / 2, 'phase3': (C + G + D) / 3}


def all_runs():
    nb = raw(pd.read_csv(RESULTS / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv'))
    rows = [{'method': 'B0_RBC', 'seed': 0, **challenge_kpis(pd.read_csv(RESULTS / 'B0_RBC' / 'eval_trace.csv'), nb)}]
    for summary in sorted(RESULTS.glob('*/seed_*/summary.json')):
        method, seed = summary.parent.parent.name, int(summary.parent.name.split('_')[1])
        trace = pd.read_csv(summary.parent / 'eval_trace.csv')
        rows.append({'method': method, 'seed': seed, **challenge_kpis(trace, nb)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    runs = all_runs()
    runs.to_csv(RESULTS / 'challenge_kpis.csv', index=False)
    table = runs.groupby('method').agg(n=('seed', 'count'), phase1=('phase1', 'mean'), phase3=('phase3', 'mean'),
                                       phase3_std=('phase3', 'std'), C=('C', 'mean'), G=('G', 'mean'), D=('D', 'mean'),
                                       R=('R', 'mean'), one_minus_L=('1-L', 'mean')).sort_values('phase3')
    pd.set_option('display.width', 200)
    print(table.round(3).to_string())
