"""Deployable model predictive control (no RL): causal forecasts + receding-horizon LP, run for one year in CityLearn.

Uses MPCController from the core (forecasts made only from past observations, so it could run in a real building).
Results go to results/<name>/seed_0/ like any other run.
Usage (from the project folder): python tools/mpc_eval.py <name> [horizon=24] [w_carbon=1] [w_grid=0] [term_frac=1]
                                 [dataset=- (phase_1)] [w_peak=0]
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import sys
import time

ns = {'__name__': '__main__'}
for c in json.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'code', '00_core.ipynb'), encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']

NAME = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_CARBON = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
W_GRID = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
TERM = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
DATASET = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else ns['DATASET_NAME']
W_PEAK = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0


def run(dataset):
    env = ns['CityLearnEnv'](schema=dataset, central_agent=True)
    obs = env.reset()
    B = len(env.buildings)
    ctrl = ns['MPCController'](B, H, W_CARBON, W_GRID, TERM, W_PEAK)
    s, k = ns['CENTRAL_SHARED_OBS'], ns['CENTRAL_OBS_PER_BUILDING']
    tracker, rows, start = ns['new_tracker'](B), [], time.time()
    while True:
        o = np.array(obs[0], dtype=np.float64)
        a = ctrl.act(env.time_step, o, [o[s + k * i + 2] for i in range(B)])
        obs, _, done, _ = env.step([a])
        t = env.time_step
        rows.append(ns['trace_row'](env, t, ns['step_terms'](env, tracker), a))
        if t % 2000 == 0:
            print(f'  step {t} ({time.time() - start:.0f} s)', flush=True)
        if done:
            break
    return pd.DataFrame(rows), env


trace, env = run(DATASET)
out = RESULTS_DIR / NAME / 'seed_0'
out.mkdir(parents=True, exist_ok=True)
trace.to_csv(out / 'eval_trace.csv', index=False)
kpis = ns['compute_kpis'](trace, env.buildings)
rbc = ns['load_rbc']()
with open(RESULTS_DIR / ns['RBC_FOLDER'] / 'summary.json') as f:
    rbc_kpis = json.load(f)['kpis']
summary = {'method': NAME, 'label': f'MPC (causal forecasts), H={H}, w_carbon={W_CARBON}, w_grid={W_GRID}, term={TERM}, w_peak={W_PEAK}',
           'seed': 0, 'kpis': kpis, 'ratios_vs_rbc': ns['kpi_ratios'](kpis, rbc['trace'], rbc_kpis)}
with open(out / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'done -> {out}', flush=True)
