"""Seed ensemble: one evaluation year where each hour's action is the mean of the deterministic actions of a method's
trained seeds (each model sees observations scaled with its own VecNormalize statistics, if any). No training.
Writes results/<name>/seed_0/ like any other run.
Usage (from the phase 3 folder): python tools/ensemble_eval.py <name> <METHOD>[:seed,seed,...] [<METHOD>[:seeds] ...]
(default seeds 42,100,2026; methods must be central, non-residual agents). Example:
  python tools/ensemble_eval.py E_M33 M33_bc_lp_grid
  python tools/ensemble_eval.py E_clones_mix M33_bc_lp_grid M38_bc_grid_big M40_bc_grid2_big_dagger:42,100
"""
import json
import sys

ns = {'__name__': '__main__'}
for c in json.load(open('00_core.ipynb', encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
NAME = sys.argv[1]
SPECS = []
for arg in sys.argv[2:]:
    method, _, seeds = arg.partition(':')
    SPECS.append((method, [int(s) for s in seeds.split(',')] if seeds else [42, 100, 2026]))
for method, _ in SPECS:
    assert ns['METHODS'][method]['agent'] == 'central' and not ns['METHODS'][method].get('residual'), method

rbc = ns['load_rbc']()
first_method, first_seeds = SPECS[0]
base = ns['make_env'](first_method, first_seeds[0], rbc)
members = []
for method, seeds in SPECS:
    cfg = ns['METHODS'][method]
    for s in seeds:
        run_dir = ns['run_dir_of'](method, s)
        algo = ns['SAC'] if cfg.get('algo') == 'sac' else ns['PPO']
        model = algo.load(run_dir / 'model.zip', device='cpu')
        scale = (lambda o: o)
        if (run_dir / 'vecnormalize.pkl').exists():
            vn = ns['VecNormalize'].load(run_dir / 'vecnormalize.pkl',
                                         ns['DummyVecEnv']([lambda: ns['make_env'](method, s, rbc)]))
            vn.training = False
            scale = vn.normalize_obs
        members.append((model, scale))

state = base.state
obs = base.policy_obs(ns['reset_state'](state, first_seeds[0]))
rows = []
while True:
    action = np.mean([m.predict(sc(obs), deterministic=True)[0] for m, sc in members], axis=0)
    cl_actions = base.to_citylearn_actions(action)
    cl_obs, _, done, terms, t = ns['env_step'](state, cl_actions)
    rows.append(ns['trace_row'](state['env'], t, terms, np.array(cl_actions[0])))
    obs = base.policy_obs(cl_obs)
    if done:
        break
trace = pd.DataFrame(rows)
name = NAME
out = RESULTS_DIR / name / 'seed_0'
out.mkdir(parents=True, exist_ok=True)
trace.to_csv(out / 'eval_trace.csv', index=False)
kpis = ns['compute_kpis'](trace, state['env'].buildings)
with open(RESULTS_DIR / ns['RBC_FOLDER'] / 'summary.json') as f:
    rbc_kpis = json.load(f)['kpis']
summary = {'method': name, 'label': 'Ensemble of ' + ', '.join(f'{m} {s}' for m, s in SPECS), 'seed': 0, 'kpis': kpis,
           'ratios_vs_rbc': ns['kpi_ratios'](kpis, rbc['trace'], rbc_kpis)}
with open(out / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'ensemble of {len(members)} models -> {out}', flush=True)
