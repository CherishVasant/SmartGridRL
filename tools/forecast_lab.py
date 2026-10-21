"""Trains the LightGBM load / PV forecasters of GBMForecaster, leave-district-out, and compares them with the causal
seasonal forecaster (CausalForecaster) on the district they will control.

Districts: train = phase_1 buildings 1-5, val = phase_2 buildings 6-10, test = phase_3 buildings 11-17. The models for a
district are trained only on the buildings of the OTHER two districts, so they never see the buildings they forecast.
All 17 buildings' observations come from one no-battery pass of citylearn_challenge_2022_phase_all (cached).
Outputs: results/forecast/<district>_load.txt, <district>_solar.txt (LightGBM boosters) and forecast_mae.csv.
Usage (from the project folder): python tools/forecast_lab.py [district ...]   (default: train val test)
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import sys
import time

import lightgbm as lgb

ns = {'__name__': '__main__'}
for c in json.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'code', '00_core.ipynb'), encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
OUT = RESULTS_DIR / 'forecast'
OUT.mkdir(parents=True, exist_ok=True)
VERSION = 2 if '--v2' in sys.argv else 1           # v2: extra features, bigger models, 3-seed ensemble
MODEL_DIR = OUT / 'v2' if VERSION == 2 else OUT
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sys.argv = [a for a in sys.argv if a != '--v2']
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 24


def all_observations():
    path = OUT / 'phase_all_observations.npy'
    if path.exists():
        return np.load(path)
    env = ns['CityLearnEnv'](schema='citylearn_challenge_2022_phase_all', central_agent=True)
    obs = env.reset()
    rows = [np.array(obs[0])]
    done = False
    while not done:
        obs, _, done, _ = env.step([np.zeros(len(env.buildings), dtype=np.float32)])
        rows.append(np.array(obs[0]))
    O = np.array(rows)
    np.save(path, O)
    return O


O = all_observations()
T = len(O)
hist = ns['GBMForecaster'](17, H, models=None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
feat = ns['GBMForecaster'].features


def dataset(x, buildings, steps):
    """Features and targets (x[t + h]) for the given buildings at the given decision times."""
    X, y = [], []
    xb = x[:, buildings]
    for t in steps:
        X.append(feat(xb, hist.weather, hist.weather_pred, hist.cal, t, H, VERSION))
        sc = ns['GBMForecaster'].scale(xb, t)
        y.append(np.array([xb[t + h] / sc for h in range(1, H + 1)]).ravel())
    return np.concatenate(X), np.concatenate(y)


def seasonal_mae(target_buildings, steps):
    """MAE of the causal seasonal forecaster's net-load forecast (load - solar), per horizon, summed over buildings."""
    fc = ns['CausalForecaster'](len(target_buildings), H, max_steps=T + 1)
    s, k = ns['CENTRAL_SHARED_OBS'], ns['CENTRAL_OBS_PER_BUILDING']
    errs = np.zeros(H)
    net = hist.load[:, target_buildings] - hist.solar[:, target_buildings]
    step_set = set(steps)
    for t in range(max(steps) + 1):
        fc.load[t], fc.solar[t] = hist.load[t, target_buildings], hist.solar[t, target_buildings]
        fc.ci[t], fc.price[t] = hist.ci[t], hist.price[t]
        if t in step_set:
            base, _, _ = fc.forecast(t)
            errs += np.abs(base - net[t + 1:t + H + 1]).sum(axis=1)
    return errs / len(steps)


rows = []
targets = sys.argv[1:] or list(DISTRICTS)
params = {'objective': 'l1', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 50, 'feature_fraction': 0.9,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1, 'num_threads': 4}
for district in targets:
    start = time.time()
    tgt = DISTRICTS[district]
    others = [b for d, bs in DISTRICTS.items() if d != district for b in bs]
    train_steps = list(range(168, T - H - 1, 2))
    eval_steps = list(range(168, T - H - 1, 6))
    preds = {}
    for name, x in [('load', hist.load), ('solar', hist.solar)]:
        Xtr, ytr = dataset(x, others, train_steps)
        Xev, _ = dataset(x, tgt, eval_steps)
        sc = np.array([ns['GBMForecaster'].scale(x[:, tgt], t) for t in eval_steps])[:, None, :]
        if VERSION == 1:
            model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=400)
            model.save_model(str(OUT / f'{district}_{name}.txt'))
            p = model.predict(Xev)
        else:
            p = 0
            for k in range(3):
                pk = dict(params, num_leaves=127, learning_rate=0.03, seed=k, bagging_seed=k, feature_fraction_seed=k)
                model = lgb.train(pk, lgb.Dataset(Xtr, ytr), num_boost_round=900)
                model.save_model(str(MODEL_DIR / f'{district}_{name}_s{k}.txt'))
                p = p + model.predict(Xev) / 3
        preds[name] = p.reshape(len(eval_steps), H, len(tgt)) * sc
        print(f'  {district} {name}: trained on {len(others)} other buildings, {len(ytr):,} rows', flush=True)
    net_fc = np.maximum(preds['load'], 0) - np.maximum(preds['solar'], 0)
    net = hist.load[:, tgt] - hist.solar[:, tgt]
    actual = np.array([net[t + 1:t + H + 1] for t in eval_steps])
    gbm = np.abs(net_fc - actual).sum(axis=2).mean(axis=0)
    seas = seasonal_mae(tgt, eval_steps)
    for h in range(H):
        rows.append({'district': district, 'h': h + 1, 'mae_gbm': gbm[h], 'mae_seasonal': seas[h]})
    print(f'{district}: district net-load MAE (kWh, summed over buildings) h=1 {gbm[0]:.3f} vs seasonal {seas[0]:.3f} | '
          f'h=6 {gbm[5]:.3f} vs {seas[5]:.3f} | h=24 {gbm[23]:.3f} vs {seas[23]:.3f} | mean {gbm.mean():.3f} vs '
          f'{seas.mean():.3f} ({time.time() - start:.0f} s)', flush=True)
pd.DataFrame(rows).to_csv(MODEL_DIR / 'forecast_mae.csv', index=False)
