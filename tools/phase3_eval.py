"""Official CityLearn Challenge 2022 Phase III scoring of a deployable controller on all three districts.

Districts: phase_1 (5 public train buildings), phase_2 (5 validation buildings), phase_3 (7 test buildings). For each,
the controller runs one year in CityLearn and C, G, D are computed against the no-battery run on the same buildings
(district-level, as in the challenge). Scores (lower is better):
    S_d     = mean(C, G, D) of district d
    public  = 0.4 S_train + 0.6 S_val                     (Phase III public leaderboard; Together 0.728)
    private = 0.2 S_train + 0.3 S_val + 0.5 S_test        (Phase III private leaderboard, used to pick the winners)
Results: results/phase3/<name>/<district>_trace.csv and results/phase3/scores.csv (one row per run).

Usage (from the project folder): python tools/phase3_eval.py <name> mpc [horizon] [w_carbon] [w_grid] [term_frac]
                                   [w_peak] [solar]
       python tools/phase3_eval.py <name> mpcgbm | mpcblend [horizon] [w_carbon] [w_grid] [term_frac]
       python tools/phase3_eval.py <name> rltuned <METHOD> [seed]      (a trained mpc_tune policy)
       python tools/phase3_eval.py <name> mpcdeep <mlp|lstm|transformer> [horizon] [w_carbon] [w_grid]
       python tools/phase3_eval.py <name> mpcfc <gbm|gbm_tuned|deep_mlp|...> [horizon] [w_carbon] [w_grid] [scenarios]
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import sys
import time
from pathlib import Path

ns = {'__name__': '__main__'}
for c in json.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'code', '00_core.ipynb'), encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
sys.path.insert(0, str(Path(__file__).parent))
import challenge_kpis as ck   # noqa: E402

DISTRICTS = {'train': 'citylearn_challenge_2022_phase_1', 'val': 'citylearn_challenge_2022_phase_2',
             'test': 'citylearn_challenge_2022_phase_3'}
OUT = RESULTS_DIR / 'phase3'


def run_year(dataset, make_policy, district=None):
    """One year on `dataset` with a central CityLearn env; make_policy(n_buildings) -> f(t, obs, soc) -> actions
    (None = no battery)."""
    env = ns['CityLearnEnv'](schema=dataset, central_agent=True)
    obs = env.reset()
    B = len(env.buildings)
    policy = make_policy(B, district) if make_policy is not None else None
    s, k = ns['CENTRAL_SHARED_OBS'], ns['CENTRAL_OBS_PER_BUILDING']
    tracker, rows = ns['new_tracker'](B), []
    while True:
        o = np.array(obs[0], dtype=np.float64)
        a = np.zeros(B, dtype=np.float32) if policy is None else policy(env.time_step, o, [o[s + k * i + 2] for i in range(B)])
        obs, _, done, _ = env.step([a])
        rows.append(ns['trace_row'](env, env.time_step, ns['step_terms'](env, tracker), a))
        if done:
            return pd.DataFrame(rows)


def no_battery(district):
    path = OUT / f'no_battery_{district}.csv'
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        run_year(DISTRICTS[district], None).to_csv(path, index=False)
    return ck.raw(pd.read_csv(path))


def score(name, make_policy, label):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    row = {'name': name, 'label': label}
    for d, dataset in DISTRICTS.items():
        start = time.time()
        trace = run_year(dataset, make_policy, d)
        trace.to_csv(out / f'{d}_trace.csv', index=False)
        k = ck.challenge_kpis(trace, no_battery(d))
        row.update({f'{d}_{m}': k[m] for m in ['C', 'G', 'D', 'phase1', 'phase3']})
        print(f"  {d}: C {k['C']:.3f} G {k['G']:.3f} D {k['D']:.3f} S {k['phase3']:.3f} ({time.time() - start:.0f} s)",
              flush=True)
    row['public'] = 0.4 * row['train_phase3'] + 0.6 * row['val_phase3']
    row['private'] = 0.2 * row['train_phase3'] + 0.3 * row['val_phase3'] + 0.5 * row['test_phase3']
    print(f"{name}: public {row['public']:.3f} (Together 0.728) | private {row['private']:.3f}", flush=True)
    log = OUT / 'scores.csv'
    pd.DataFrame([row]).to_csv(log, mode='a', header=not log.exists(), index=False)
    return row


if __name__ == '__main__':
    NAME, KIND = sys.argv[1], sys.argv[2]
    if KIND == 'mpc':
        H = int(sys.argv[3]) if len(sys.argv) > 3 else 24
        WC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        WG = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        TF = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        WP = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
        SOLAR = len(sys.argv) > 8 and sys.argv[8] == 'solar'
        make = lambda B, d: ns['MPCController'](B, H, WC, WG, TF, WP, SOLAR).act
        score(NAME, make, f'MPC H={H} w_carbon={WC} w_grid={WG} term={TF} w_peak={WP} solar_model={SOLAR}')
    elif KIND == 'mpcgbm':
        # MPC with the leave-district-out LightGBM load / PV forecasters of tools/forecast_lab.py
        import lightgbm as lgb
        H = int(sys.argv[3]) if len(sys.argv) > 3 else 24
        WC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        WG = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
        TF = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        def make(B, d):
            models = {n: lgb.Booster(model_file=str(RESULTS_DIR / 'forecast' / f'{d}_{n}.txt')) for n in ['load', 'solar']}
            fc = ns['GBMForecaster'](B, H, models)
            return ns['MPCController'](B, H, WC, WG, TF, forecaster=fc).act
        score(NAME, make, f'MPC + LightGBM forecasts H={H} w_carbon={WC} w_grid={WG} term={TF}')
    elif KIND == 'mpcblend':
        # MPC with the online blend of seasonal and LightGBM forecasts (BlendForecaster)
        import lightgbm as lgb
        H = int(sys.argv[3]) if len(sys.argv) > 3 else 24
        WC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        WG = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
        TF = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        def make(B, d):
            models = {n: lgb.Booster(model_file=str(RESULTS_DIR / 'forecast' / f'{d}_{n}.txt')) for n in ['load', 'solar']}
            fc = ns['BlendForecaster'](B, H, models)
            return ns['MPCController'](B, H, WC, WG, TF, forecaster=fc).act
        score(NAME, make, f'MPC + blended seasonal/LightGBM forecasts H={H} w_carbon={WC} w_grid={WG} term={TF}')
    elif KIND == 'mpcdeep':
        # MPC with the deep (Colab-trained, TorchScript) forecasters: argv[3] = arch (mlp | lstm | transformer)
        ARCH = sys.argv[3]
        H = int(sys.argv[4]) if len(sys.argv) > 4 else 24
        WC = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        WG = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
        def make(B, d):
            fc = ns['DeepForecaster'](B, H, ns['load_deep_models'](d, ARCH))
            return ns['MPCController'](B, H, WC, WG, 1.0, forecaster=fc).act
        score(NAME, make, f'MPC + deep {ARCH} forecasts H={H} w_carbon={WC} w_grid={WG}')
    elif KIND == 'mpcfc':
        # MPC with any forecaster of make_mpc_controller: gbm | deep_<arch> | gbm+deep_<arch>
        FC = sys.argv[3]
        cfg = {'horizon': int(sys.argv[4]) if len(sys.argv) > 4 else 24,
               'w_carbon': float(sys.argv[5]) if len(sys.argv) > 5 else 1.0,
               'w_grid': float(sys.argv[6]) if len(sys.argv) > 6 else 3.0, 'forecaster': FC,
               'scenarios': int(sys.argv[7]) if len(sys.argv) > 7 else 0}
        make = lambda B, d: ns['make_mpc_controller'](cfg, B, d).act
        score(NAME, make, f"MPC + {FC} forecasts H={cfg['horizon']} w_carbon={cfg['w_carbon']} w_grid={cfg['w_grid']}"
              + (f" scenarios={cfg['scenarios']}" if cfg['scenarios'] > 1 else ''))
    elif KIND == 'rltuned':
        # RL-tuned MPC: a trained mpc_tune policy (district-level observations) sets the MPC weights every hour
        from types import SimpleNamespace
        METHOD, SEED = sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 42
        holder = ns['CentralEnv'](METHOD, SEED, ns['load_rbc']())
        model = ns['algo_class'](METHOD).load(ns['run_dir_of'](METHOD, SEED) / 'model.zip', device='cpu')
        cfg = ns['METHODS'][METHOD]['mpc_tune']
        def make(B, d):
            holder.ctrl, holder.prev_fc = ns['make_mpc_controller'](cfg, B, d), None
            holder.env = SimpleNamespace(time_step=0, buildings=[None] * B)
            def policy(t, o, soc):
                holder.env.time_step = t
                a, _ = model.predict(holder.mpc_obs([o]), deterministic=True)
                return holder.to_citylearn_actions(a)[0]
            return policy
        score(NAME, make, f'RL-tuned MPC {METHOD} seed {SEED}')
