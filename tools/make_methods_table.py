"""One row per method with every metric, written to all_methods_results.csv and .xlsx in the project folder.

Sources (all in results/): challenge_kpis.csv (official CityLearn 2022 KPIs on the 5 training buildings, per seed),
<method>/seed_*/summary.json (KPIs relative to the RBC), transfer/transfer_kpis.csv (the same policies on the 5 unseen
validation buildings) and phase3/scores.csv (official Phase III public / private scores over all 17 buildings).
All KPIs are ratios (to no battery unless the column says RBC); lower is better. Numeric columns are means over the
seeds listed in the last column.
Usage (from the project folder): python tools/make_methods_table.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results'


def method_type(m):
    """(type, deployable) of a method."""
    num = int(re.match(r'[A-Z]+(\d+)', m).group(1)) if re.match(r'[MB]\d+', m) else None
    if m == 'B5_no_battery':
        return 'Reference: no battery', 'yes'
    if m == 'B0_RBC':
        return 'Reference: rule-based controller', 'yes'
    if m.startswith('B'):
        return 'Baseline RL (PPO, phase 2 rewards)', 'yes'
    if m.startswith('O'):
        return 'Perfect-foresight LP plan (upper bound)', 'no (uses future data)'
    if m.startswith('ER_'):
        return 'Ensemble of RL policies', 'yes'
    if m.startswith('E_'):
        return 'Ensemble of plan-imitation clones', 'no (imitates a foresight plan)'
    if m.startswith('MPC') or m.startswith('P3_MPC'):
        return 'MPC with causal seasonal forecasts (no RL)', 'yes'
    if m.startswith('P3_GBMMPC'):
        return 'MPC with LightGBM forecasts (no RL)', 'yes'
    if m.startswith('P3_BLEND'):
        return 'MPC with blended forecasts (no RL)', 'yes'
    if num is not None:
        if 29 <= num <= 32:
            return 'Residual RL on a perfect-foresight plan', 'no (uses future data)'
        if 33 <= num <= 89:
            return 'Imitation (cloning) of a perfect-foresight plan', 'no (imitates a foresight plan)'
        if 129 <= num <= 135:
            return 'RL-tuned MPC (forecasting + optimisation + RL)', 'yes'
        if num <= 28 and num <= 6:
            return 'RL (PPO, Stage A rewards)', 'yes'
        return 'RL (model-free: PPO / SAC / TQC / CrossQ)', 'yes'
    return 'Other', ''


def order_key(m):
    prefix = re.match(r'[A-Za-z_]+?(?=\d)|[A-Za-z_]+', m).group(0)
    num = re.search(r'\d+', m)
    rank = {'B': 0, 'M': 1, 'MPC': 2, 'P3_MPC_g': 3, 'P3_GBMMPC_g': 4, 'P3_BLEND_g': 5, 'ER_M': 6, 'E_': 7, 'O': 8}
    return rank.get(prefix, 9), int(num.group(0)) if num else 0, m


def seeds_text(seeds):
    return ', '.join(str(s) for s in sorted(seeds, key=int))


# ---- official KPIs on the training buildings, per method x seed
kpi = pd.read_csv(R / 'challenge_kpis.csv')
rows = {}
for m, g in kpi.groupby('method'):
    row = {'method': m}
    for col, name in [('phase1', 'phase1_score'), ('phase3', 'phase3_score')]:
        row[f'{name}_mean'] = g[col].mean()
        row[f'{name}_std'] = g[col].std(ddof=1) if len(g) > 1 else np.nan
    for col, name in [('C', 'cost_C'), ('G', 'carbon_G'), ('R', 'ramping_R'), ('1-L', 'one_minus_load_factor'),
                      ('D', 'grid_D')]:
        row[name] = g[col].mean()
    row['n_seeds'] = len(g)
    row['seeds_averaged'] = seeds_text(g['seed'].astype(int).tolist())
    rows[m] = row

# ---- labels, training length and KPIs relative to the RBC (summary.json)
RBC_KEYS = ['score', 'cost', 'carbon', 'ramping', 'daily_peak', 'annual_peak', 'one_minus_load_factor',
            'carbon_per_kwh_imported', 'battery_cycles', 'self_consumption']
for m, row in rows.items():
    summaries = [json.load(open(p)) for p in sorted((R / m).glob('seed_*/summary.json'))]
    if not summaries and m == 'B0_RBC':
        row['label'] = 'Rule-based controller (CityLearn BasicRBC)'
        continue
    if not summaries:
        continue
    row['label'] = summaries[0].get('label', '')
    row['env_steps_trained'] = summaries[0].get('env_steps_trained', np.nan)
    for k in RBC_KEYS:
        vals = [s['ratios_vs_rbc'][k] for s in summaries if k in s.get('ratios_vs_rbc', {})]
        if vals:
            row[f'{k}_vs_RBC'] = float(np.mean(vals))
    cycles = [s['kpis']['battery_cycles'] for s in summaries if 'battery_cycles' in s.get('kpis', {})]
    if cycles:
        row['battery_cycles_per_year'] = float(np.mean(cycles))
if 'B5_no_battery' in rows:
    rows['B5_no_battery']['label'] = 'No battery (reference: all KPIs = 1)'

# ---- unseen validation buildings (phase_2), same policies unchanged
tr_path = R / 'transfer' / 'transfer_kpis.csv'
if tr_path.exists():
    tr = pd.read_csv(tr_path).drop_duplicates(['method', 'seed'], keep='last')
    for m, g in tr.groupby('method'):
        row = rows.setdefault(m, {'method': m})
        for col, name in [('val_C', 'unseen_val_cost_C'), ('val_G', 'unseen_val_carbon_G'), ('val_D', 'unseen_val_grid_D'),
                          ('val_phase3', 'unseen_val_phase3_score'), ('public_score', 'public_score_0.4train_0.6val')]:
            row[name] = g[col].mean()
        row['unseen_val_seeds'] = seeds_text(g['seed'].astype(int).tolist())

# ---- official Phase III scores over all 17 buildings (train / val / test districts)
p3_path = R / 'phase3' / 'scores.csv'
if p3_path.exists():
    p3 = pd.read_csv(p3_path).drop_duplicates('name', keep='last')
    for _, r in p3.iterrows():
        name = 'P3_MPC_g0' if r['name'] == 'NB_cache_dummy' else r['name']
        rl = re.match(r'P3_RLTUNED_(M\d+)', name)
        method = next((m for m in rows if rl and m.startswith(rl.group(1) + '_')), None) if rl else name
        row = rows.setdefault(method, {'method': method, 'label': r['label']})
        if not rl:   # controllers scored only here: their training-district KPIs come from the Phase III run
            row.update({'phase1_score_mean': r['train_phase1'], 'phase3_score_mean': r['train_phase3'],
                        'cost_C': r['train_C'], 'carbon_G': r['train_G'], 'grid_D': r['train_D'], 'n_seeds': 1,
                        'seeds_averaged': 'deterministic (no seed)'})
        for d in ['train', 'val', 'test']:
            row[f'phase3_{d}_district_score'] = r[f'{d}_phase3']
        row['phase3_test_cost_C'], row['phase3_test_carbon_G'], row['phase3_test_grid_D'] = r['test_C'], r['test_G'], r['test_D']
        row['PHASE3_PUBLIC_score'], row['PHASE3_PRIVATE_score'] = r['public'], r['private']
        if rl:
            row['phase3_seed'] = '42'

df = pd.DataFrame(rows.values())
for m in df['method']:
    pass
types = df['method'].map(method_type)
df.insert(2, 'type', [t[0] for t in types])
df.insert(3, 'deployable', [t[1] for t in types])
for m in df.index:
    if df.at[m, 'seeds_averaged'] != df.at[m, 'seeds_averaged']:   # NaN
        df.at[m, 'seeds_averaged'] = ''
# deterministic controllers (plans, MPC, RBC, ensembles) are stored as seed 0
det = df['method'].str.match(r'^(O\d|MPC|E_|ER_|B0_|B5_)')
df.loc[det & (df['seeds_averaged'] == '0'), 'seeds_averaged'] = 'deterministic (no seed)'
df = df.iloc[sorted(range(len(df)), key=lambda i: order_key(df['method'].iloc[i]))]
first = ['method', 'label', 'type', 'deployable', 'env_steps_trained', 'n_seeds',
         'phase1_score_mean', 'phase1_score_std', 'phase3_score_mean', 'phase3_score_std',
         'cost_C', 'carbon_G', 'ramping_R', 'one_minus_load_factor', 'grid_D']
p3_cols = ['PHASE3_PUBLIC_score', 'PHASE3_PRIVATE_score', 'phase3_train_district_score', 'phase3_val_district_score',
           'phase3_test_district_score', 'phase3_test_cost_C', 'phase3_test_carbon_G', 'phase3_test_grid_D', 'phase3_seed']
unseen = ['unseen_val_cost_C', 'unseen_val_carbon_G', 'unseen_val_grid_D', 'unseen_val_phase3_score',
          'public_score_0.4train_0.6val', 'unseen_val_seeds']
rbc_cols = [c for c in df.columns if c.endswith('_vs_RBC')] + ['battery_cycles_per_year']
cols = [c for c in first + p3_cols + unseen + rbc_cols if c in df.columns]
cols += [c for c in df.columns if c not in cols and c != 'seeds_averaged'] + ['seeds_averaged']
df = df[cols]
num = df.select_dtypes('number').columns
df[num] = df[num].round(4)

df.to_csv(ROOT / 'all_methods_results.csv', index=False)
with pd.ExcelWriter(ROOT / 'all_methods_results.xlsx', engine='openpyxl') as xl:
    df.to_excel(xl, sheet_name='All methods', index=False)
    notes = pd.DataFrame({'column / term': [
        'all KPIs', 'phase1_score', 'phase3_score', 'cost_C / carbon_G', 'ramping_R / one_minus_load_factor / grid_D',
        'PHASE3_PUBLIC_score', 'PHASE3_PRIVATE_score', 'unseen_val_*', '*_vs_RBC', 'deployable', 'seeds_averaged'],
        'meaning': [
        'Ratios to the no-battery case on the 5 training buildings (citylearn_challenge_2022_phase_1); lower is better',
        'mean(C, G): the challenge Phase I leaderboard metric',
        'mean(C, G, D): the challenge Phase III metric for one district',
        'District electricity cost and carbon emissions (exports earn nothing), relative to no battery',
        'Ramping, 1 - monthly load factor, and D = mean(R, 1-L)',
        '0.4 x S(train district) + 0.6 x S(validation district); published #1 Together = 0.728',
        '0.2 x S(train) + 0.3 x S(validation) + 0.5 x S(7 test buildings): the winner-selection score',
        'The same trained policy, unchanged, on the 5 unseen validation buildings (phase_2)',
        'KPIs relative to the CityLearn rule-based controller (older reference)',
        'no = uses future data (perfect-foresight plan or its imitation): a reference, not a real-world controller',
        'Seeds whose results were averaged for the numeric columns (Phase III columns: phase3_seed)']})
    notes.to_excel(xl, sheet_name='Column notes', index=False)
    for ws in xl.book.worksheets:
        ws.freeze_panes = 'B2'
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
print(f'{len(df)} methods x {len(df.columns)} columns -> all_methods_results.csv / .xlsx')
