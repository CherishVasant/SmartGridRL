"""Perfect-foresight linear-programming battery schedule for the official CityLearn 2022 objective (a reference, not RL).

Knows the whole year in advance (like the Phase I leaderboard, which scored on the training buildings), so it shows how
much headroom exists and can serve as a teacher for imitation. The LP plans grid-side battery energy per building and
hour; the plan is then replayed in CityLearn (tracking the planned state of charge), so its KPIs come from the real
simulator. Results go to results/<name>/seed_0/ like any other run.

Objective (all ÷ no-battery totals): C + G + w_grid * (R + P) / 2, where R = ramping and P = mean monthly peak
(a convex proxy for 1 - load factor). w_grid = 0 targets the Phase I score mean(C, G).
Usage (from the phase 3 folder): python tools/lp_oracle.py <w_grid> <name> [w_carbon=1] [efficiency=0.9]
(efficiency = the battery's one-cycle efficiency; CityLearn's curve gives 0.83-0.9 depending on power)
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ns = {'__name__': '__main__'}
for c in json.load(open('00_core.ipynb', encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
pd, RESULTS_DIR = ns['pd'], ns['RESULTS_DIR']
W_GRID, NAME = float(sys.argv[1]), sys.argv[2]
W_CARBON = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
EFFICIENCY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9

# ---- Data: no-battery net load per building (exogenous), price and carbon intensity per simulated step
nb = pd.read_csv(RESULTS_DIR / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv')
steps = nb['step'].values                      # 1 .. T: the step each action produces
B = 5
base = nb[[f'load_b{i}' for i in range(B)]].values           # (T, B) kWh
price, ci = nb['pricing'].values, nb['carbon_intensity'].values
T = len(steps)
CAP, PMAX, ETA = 6.4, 5.0, EFFICIENCY ** 0.5    # per battery; CityLearn applies sqrt(efficiency) each way
c_nb = np.maximum(0, base.sum(1) * price).sum(); g_nb = np.maximum(0, base.sum(1) * ci).sum()
r_nb = np.abs(np.diff(base.sum(1))).sum()
months = [np.arange(i, min(i + 730, T)) for i in range(0, T, 730) if min(i + 730, T) - i >= 24]
p_nb = np.mean([base.sum(1)[m].max() for m in months])

# ---- Variables: ch[b,t], dis[b,t] (grid-side kWh), soc[b,t] (kWh stored after step t), imp[t] >= max(0, E_t),
#      rp[t] >= |E_t - E_{t-1}|, pk[m] >= E_t for t in month m
nv = 0
def block(n):
    global nv
    idx = np.arange(nv, nv + n); nv += n; return idx
ch, dis, soc = block(B * T).reshape(B, T), block(B * T).reshape(B, T), block(B * T).reshape(B, T)
imp, rp, pk = block(T), block(T - 1), block(len(months))
cost = np.zeros(nv)
cost[imp] = price / c_nb + W_CARBON * ci / g_nb
cost[rp] = W_GRID / 2 / r_nb
cost[pk] = W_GRID / 2 / p_nb / len(months)

rows, cols, vals, rhs_ub = [], [], [], []
eq_rows, eq_cols, eq_vals, rhs_eq = [], [], [], []
r = 0
def ub(entries, b):
    global r
    for col, v in entries:
        rows.append(r); cols.append(col); vals.append(v)
    rhs_ub.append(b); r += 1
# E_t = sum_b base + ch - dis  (vectorised below as sparse blocks)
def E_terms(t, sign=1.0):
    return [(ch[b, t], sign) for b in range(B)] + [(dis[b, t], -sign) for b in range(B)]
# SoC dynamics (equalities): soc[t] = soc[t-1] + ETA*ch[t] - dis[t]/ETA, soc[-1] = 0
q = 0
for b in range(B):
    for t in range(T):
        eq_rows += [q, q, q]; eq_cols += [soc[b, t], ch[b, t], dis[b, t]]; eq_vals += [1.0, -ETA, 1.0 / ETA]
        if t > 0:
            eq_rows.append(q); eq_cols.append(soc[b, t - 1]); eq_vals.append(-1.0)
        rhs_eq.append(0.0); q += 1
for t in range(T):
    s_base = base[t].sum()
    ub(E_terms(t) + [(imp[t], -1.0)], -s_base)                       # imp >= E_t
    if t > 0:
        d_base = s_base - base[t - 1].sum()
        ub(E_terms(t) + E_terms(t - 1, -1.0) + [(rp[t - 1], -1.0)], -d_base)   # rp >= E_t - E_{t-1}
        ub(E_terms(t, -1.0) + E_terms(t - 1) + [(rp[t - 1], -1.0)], d_base)    # rp >= E_{t-1} - E_t
    for b in range(B):
        # power taper above 80% SoC (CityLearn capacity_power_curve): ch, dis <= PMAX * (1 - 4 * (soc_prev/CAP - 0.8))
        prev = [(soc[b, t - 1], 4 * PMAX / CAP)] if t > 0 else []
        ub([(ch[b, t], 1.0)] + prev, 4.2 * PMAX if t > 0 else 4.2 * PMAX)
        ub([(dis[b, t], 1.0)] + prev, 4.2 * PMAX)
if W_GRID > 0:
    for k, m in enumerate(months):
        for t in m:
            ub(E_terms(t) + [(pk[k], -1.0)], -base[t].sum())          # pk >= E_t
A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
A_eq = sp.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(q, nv))
bounds = np.zeros((nv, 2)); bounds[:, 1] = np.inf
bounds[ch.ravel(), 1] = PMAX; bounds[dis.ravel(), 1] = PMAX; bounds[soc.ravel(), 1] = CAP
start = time.time()
print(f'LP: {nv:,} variables, {r:,} inequalities, {q:,} equalities, w_grid = {W_GRID}', flush=True)
res = linprog(cost, A_ub=A_ub, b_ub=rhs_ub, A_eq=A_eq, b_eq=rhs_eq, bounds=bounds, method='highs')
print(f'  {res.message} ({time.time() - start:.0f} s), planned objective {res.fun:.4f}', flush=True)
plan_soc = res.x[soc.ravel()].reshape(B, T)

# ---- Replay in CityLearn: each hour, request the energy that moves the battery to the planned SoC
env = ns['CityLearnEnv'](schema=ns['DATASET_NAME'], central_agent=False)
env.reset()
tracker, rows_out = ns['new_tracker'](B), []
for i in range(T):
    actions = []
    for b, bld in enumerate(env.buildings):
        # after a step CityLearn stores the new state of charge at the current time_step index
        cur = bld.electrical_storage.soc[bld.electrical_storage.time_step] * CAP
        delta = plan_soc[b, i] - cur
        energy = delta / ETA if delta >= 0 else delta * ETA          # grid-side energy for that SoC change
        actions.append([float(np.clip(energy / CAP, -0.78125, 0.78125))])
    _, _, done, _ = env.step(actions)
    t = env.time_step
    rows_out.append(ns['trace_row'](env, t, ns['step_terms'](env, tracker), [a[0] for a in actions]))
    if done:
        break
trace = pd.DataFrame(rows_out)
out = RESULTS_DIR / NAME / 'seed_0'
out.mkdir(parents=True, exist_ok=True)
trace.to_csv(out / 'eval_trace.csv', index=False)
kpis = ns['compute_kpis'](trace, env.buildings)
rbc = ns['load_rbc']()
with open(RESULTS_DIR / ns['RBC_FOLDER'] / 'summary.json') as f:
    rbc_kpis = json.load(f)['kpis']
summary = {'method': NAME, 'label': f'LP oracle (perfect foresight), w_grid={W_GRID}, w_carbon={W_CARBON}, eff={EFFICIENCY}',
           'seed': 0, 'kpis': kpis,
           'ratios_vs_rbc': ns['kpi_ratios'](kpis, rbc['trace'], rbc_kpis), 'planned_objective': float(res.fun)}
with open(out / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'  replayed in CityLearn: {len(trace)} steps -> {out}', flush=True)
