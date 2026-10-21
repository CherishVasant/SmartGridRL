"""Generalisation test: runs trained policies, unchanged, on the CityLearn Challenge 2022 held-out buildings.

The challenge picked its winners on buildings the controllers never saw. `citylearn_challenge_2022_phase_2` holds the
5 validation buildings (same observation / action layout as our 5 training buildings), so a central policy trained on
phase_1 runs on it as is. Official KPIs (C, G, D) are computed against the no-battery run on the same buildings.
The Phase III public leaderboard score was 0.4 x mean(C, G, D) on the training buildings + 0.6 x mean(C, G, D) on the
validation buildings, so `public_score` here is directly comparable with it (Together 0.728).

Writes results/transfer/<dataset>/<method>_seed_<seed>_trace.csv and appends to results/transfer/transfer_kpis.csv.
Usage (from the project folder): python tools/transfer_eval.py <dataset> <METHOD>[:seed,seed] [<METHOD>[:seeds] ...]
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import sys
from pathlib import Path

ns = {'__name__': '__main__'}
for c in json.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'code', '00_core.ipynb'), encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
sys.path.insert(0, str(Path(__file__).parent))
import challenge_kpis as ck   # noqa: E402

DATASET = sys.argv[1]
SPECS = []
for arg in sys.argv[2:]:
    method, _, seeds = arg.partition(':')
    SPECS.append((method, [int(s) for s in seeds.split(',')] if seeds else [42, 100, 2026]))
OUT = RESULTS_DIR / 'transfer' / DATASET
OUT.mkdir(parents=True, exist_ok=True)
rbc = ns['load_rbc']()   # phase_1 reference: only used to build the method's reward state, never for scoring


def run_trace(env_obj, policy):
    """One deterministic year on env_obj (CentralEnv); policy(obs) -> action (None = no battery)."""
    state = env_obj.state
    obs = env_obj.policy_obs(ns['reset_state'](state, 0))
    rows = []
    while True:
        action = np.zeros(len(state['env'].buildings), dtype=np.float32) if policy is None else policy(obs)
        cl_actions = env_obj.to_citylearn_actions(action) if policy is not None else [action]
        cl_obs, _, done, terms, t = ns['env_step'](state, cl_actions)
        rows.append(ns['trace_row'](state['env'], t, terms, np.array(cl_actions[0])))
        obs = env_obj.policy_obs(cl_obs)
        if done:
            return pd.DataFrame(rows)


def make_env_on(method, seed):
    ns['DATASET_NAME'] = DATASET
    try:
        return ns['CentralEnv'](method, seed, rbc)
    finally:
        ns['DATASET_NAME'] = 'citylearn_challenge_2022_phase_1'


# No-battery reference on the held-out buildings (cached)
nb_path = OUT / 'no_battery_trace.csv'
if not nb_path.exists():
    run_trace(make_env_on(SPECS[0][0], 0), None).to_csv(nb_path, index=False)
nb_raw = ck.raw(pd.read_csv(nb_path))
nb_train = ck.raw(pd.read_csv(RESULTS_DIR / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv'))

rows = []
for method, seeds in SPECS:
    cfg = ns['METHODS'][method]
    assert cfg['agent'] == 'central' and not cfg.get('residual'), method
    for s in seeds:
        run_dir = ns['run_dir_of'](method, s)
        model = ns['algo_class'](method).load(run_dir / 'model.zip', device='cpu')
        base = make_env_on(method, s)
        scale = lambda o: o
        if (run_dir / 'vecnormalize.pkl').exists():
            vn = ns['VecNormalize'].load(run_dir / 'vecnormalize.pkl', ns['DummyVecEnv']([lambda: base]))
            vn.training = False
            scale = vn.normalize_obs
        trace = run_trace(base, lambda o: model.predict(scale(o), deterministic=True)[0])
        trace.to_csv(OUT / f'{method}_seed_{s}_trace.csv', index=False)
        val = ck.challenge_kpis(trace, nb_raw)
        train = ck.challenge_kpis(pd.read_csv(run_dir / 'eval_trace.csv'), nb_train)
        row = {'dataset': DATASET, 'method': method, 'seed': s, 'n_buildings': len(base.env.buildings),
               **{f'val_{k}': v for k, v in val.items()}, 'train_phase3': train['phase3'],
               'public_score': 0.4 * train['phase3'] + 0.6 * val['phase3']}
        rows.append(row)
        print(f"{method} seed {s}: held-out C {val['C']:.3f} G {val['G']:.3f} D {val['D']:.3f} | held-out mean(C,G,D) "
              f"{val['phase3']:.3f} | train {train['phase3']:.3f} | 0.4/0.6 public score {row['public_score']:.3f}", flush=True)

log = RESULTS_DIR / 'transfer' / 'transfer_kpis.csv'
pd.DataFrame(rows).to_csv(log, mode='a', header=not log.exists(), index=False)
