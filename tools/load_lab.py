"""Load-forecasting lab: better forecasts of the DISTRICT TOTAL load, the quantity the ramping and load-factor KPIs use.

tools/load_diagnostics.py showed that per-building load is mostly noise (accuracy ~58% even with LightGBM, ~62% one hour
ahead), while the district total is ~75% accurate, and the per-building LightGBM summed over the district barely beats
the causal seasonal forecast on it. Experiments, all leave-district-out (nothing from the target district is used for
training or for choosing anything), scored on the target district's total on the forecast_lab.py evaluation hours:
  sum_gbm     current tuned per-building LightGBM, summed                                  (baseline)
  seasonal    causal seasonal forecast, summed                                             (baseline)
  direct      LightGBM trained directly on district totals; training series are sums of random subsets of the other
              districts' buildings ("pseudo-districts", many more series than the two real districts)
  direct_x    as direct with extra features (more recent lags, short rolling means, daily change, 4-week same-hour
              mean, 24 h mean temperature)
  blend       weighted average of direct_x, sum_gbm and seasonal; weights fitted on the training pseudo-districts
Usage (from the project folder): python tools/load_lab.py [n_subsets=40] [threads=4]
Outputs: results/forecast/load_lab/<district>_total_direct_x.txt, load_lab.csv
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import os
import sys
import time

import lightgbm as lgb

ns = {'__name__': '__main__'}
for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', '00_core.ipynb'),
                        encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
G = ns['GBMForecaster']
FC = RESULTS_DIR / 'forecast'
OUT = FC / 'load_lab'
OUT.mkdir(parents=True, exist_ok=True)
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 24
N_SUBSETS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

O = np.load(FC / 'phase_all_observations.npy')
T = len(O)
hist = G(17, H, None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
LOAD = hist.load[:T]
TEMP = hist.weather[:T, 0]
train_steps = list(range(168 * 4, T - H - 1, 3))     # 4 weeks of history for the 4-week mean
eval_steps = list(range(168, T - H - 1, 6))           # same hours as forecast_lab.py / load_diagnostics.py


def extra_features(x, t):
    """(H, 11) extra rows for one scaled series x (T, 1) at time t (GBMForecaster.total_extra, as used in control)."""
    return G.total_extra(x[:, 0], TEMP, t, H)


def series_rows(x, steps, extra):
    """Features, scaled targets and scales for one series x (T, 1)."""
    X, y, sc = [], [], []
    for t in steps:
        s = G.scale(x, t)
        f = G.features(x, hist.weather, hist.weather_pred, hist.cal, t, H)
        if extra:
            f = np.hstack([f, extra_features(x / s, t)])
        X.append(f)
        y.append(x[t + 1:t + H + 1, 0] / s[0])
        sc.append(np.full(H, s[0]))
    return np.concatenate(X), np.concatenate(y), np.concatenate(sc)


def pseudo_districts(buildings, n, rng):
    """n random subsets (size 4 .. 8, with replacement across subsets) of the given buildings, summed."""
    out = []
    for _ in range(n):
        k = rng.integers(4, min(8, len(buildings)) + 1)
        out.append(LOAD[:, rng.choice(buildings, k, replace=False)].sum(1, keepdims=True))
    return out


def seasonal_total(x, steps):
    out = []
    for t in steps:
        err = x[t] - hist._seasonal(x, t, 0)
        out.append(np.maximum([hist._seasonal(x, t, h) + err * 0.8 ** h for h in range(1, H + 1)], 0).sum(-1))
    return np.stack(out)                                                          # (N, H)


def sum_gbm_total(x, steps, district):
    model = lgb.Booster(model_file=str(FC / 'tuned' / f'{district}_load.txt'))
    out = []
    for t in steps:
        X = G.features(x, hist.weather, hist.weather_pred, hist.cal, t, H)
        out.append((np.maximum(model.predict(X), 0).reshape(H, x.shape[1]) * G.scale(x, t)).sum(-1))
    return np.stack(out)


def acc(pred, actual):
    return 1 - np.abs(pred - actual).mean() / np.abs(actual).mean()


PARAMS = {'objective': 'l1', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200,
          'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
          'verbose': -1, 'num_threads': THREADS, 'seed': 0}
ROUNDS = 600

rows = []
for district, target in DISTRICTS.items():
    start = time.time()
    rng = np.random.default_rng(0)
    others = [b for d, bs in DISTRICTS.items() if d != district for b in bs]
    train_series = pseudo_districts(others, N_SUBSETS, rng)
    actual_tot = LOAD[:, target].sum(1, keepdims=True)
    A = np.stack([actual_tot[t + 1:t + H + 1, 0] for t in eval_steps])           # (N, H)
    preds = {'seasonal': seasonal_total(LOAD[:, target], eval_steps),
             'sum_gbm': sum_gbm_total(LOAD[:, target], eval_steps, district)}
    fitted = {}
    for name, extra in [('direct', False), ('direct_x', True)]:
        parts = [series_rows(s, train_steps, extra) for s in train_series]
        Xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        model = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=ROUNDS)
        model.save_model(str(OUT / f'{district}_total_{name}.txt'))
        Xe, _, se = series_rows(actual_tot, eval_steps, extra)
        preds[name] = (np.maximum(model.predict(Xe), 0) * se).reshape(len(eval_steps), H)
        fitted[name] = model
        print(f'  {district} {name}: trained on {len(Xtr):,} rows in {(time.time() - start) / 60:.0f} min', flush=True)

    # blend weights fitted on a few training pseudo-districts only (non-negative least absolute error, coarse grid)
    val_series = pseudo_districts(others, 6, np.random.default_rng(1))
    comps = {'direct_x': [], 'seasonal': [], 'actual': []}
    for s in val_series:
        Xv, _, sv = series_rows(s, eval_steps, True)
        comps['direct_x'].append((np.maximum(fitted['direct_x'].predict(Xv), 0) * sv).reshape(len(eval_steps), H))
        comps['seasonal'].append(seasonal_total(s, eval_steps))
        comps['actual'].append(np.stack([s[t + 1:t + H + 1, 0] for t in eval_steps]))
    best_w, best_e = 1.0, np.inf
    for w in np.linspace(0, 1, 21):
        e = np.mean([np.abs(w * a + (1 - w) * b - c).mean()
                     for a, b, c in zip(comps['direct_x'], comps['seasonal'], comps['actual'])])
        if e < best_e:
            best_w, best_e = w, e
    preds['blend'] = best_w * preds['direct_x'] + (1 - best_w) * preds['seasonal']

    for name, p in preds.items():
        row = {'district': district, 'forecast': name, 'total_acc': acc(p, A)}
        for h in [1, 2, 3, 6, 12, 24]:
            row[f'h{h}'] = acc(p[:, h - 1], A[:, h - 1])
        if name == 'blend':
            row['note'] = f'w_direct_x={best_w:.2f}'
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'load_lab.csv', index=False)
    print(f'{district} done in {(time.time() - start) / 60:.0f} min', flush=True)

pd.set_option('display.width', 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
