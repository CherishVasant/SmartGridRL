"""Load-forecasting lab, round 2: improving the direct district-total model of tools/load_lab.py ('direct_x').

Same protocol (leave-district-out; training series are pseudo-districts = sums of random subsets of the OTHER districts'
buildings; scored on the target district's real total, forecast_lab.py evaluation hours). One change per variant:
  A  direct_x as saved by load_lab.py (40 pseudo-districts, every 3rd hour)                    (reference)
  B  more data: 45 pseudo-districts, every 2nd hour (vs 40 every 3rd; float32 rows, ~0.6 GB)
  C  B + more features (GBMForecaster.total_extra2): level vs the usual profile, yesterday's deviation from the profile
     at the target hour, more lags, 12 h mean, 24 h range, temperature now vs 24 h ago and vs its forecast
  D  C, 3-seed ensemble (different pseudo-districts and bagging seeds)
  E  C with LightGBM settings chosen by inner CV between the two other districts (fit on pseudo-districts of one,
     validate on pseudo-districts of the other, both ways); the target district is never used to choose
Usage (from the project folder): python tools/load_lab2.py [threads=4] [variants=ABCDE]
Outputs: results/forecast/load_lab/<district>_total_<variant>.txt, load_lab2.csv
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
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 24
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
VARIANTS = sys.argv[2] if len(sys.argv) > 2 else 'ABCDE'

O = np.load(FC / 'phase_all_observations.npy')
T = len(O)
hist = G(17, H, None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
LOAD = hist.load[:T]
eval_steps = list(range(168, T - H - 1, 6))


def total_extra2(s, weather, weather_pred, t, H):
    """(H, 11 + 10) features: total_extra plus round-2 features, for the scaled total s (1-D, known up to t)."""
    hs = np.arange(1, H + 1)
    base = G.total_extra(s, weather[:, 0], t, H)
    prof = lambda u: np.mean([s[max(u - 24 * d, 0)] for d in range(1, 29)])       # usual level at hour u (28 days)
    level_now = s[max(t - 2, 0):t + 1].mean() / (np.mean([prof(t - k) for k in range(3)]) + 1e-3)
    dev_yday = np.array([s[min(t + h - 24, t)] - prof(min(t + h - 24, t)) for h in hs])
    temp_fut = np.where(hs <= 6, weather_pred[t, 0, 0], np.where(hs <= 12, weather_pred[t, 0, 1], weather_pred[t, 0, 2]))
    extra = [level_now, s[max(t - 4, 0)], s[max(t - 8, 0)], s[max(t - 12, 0)], s[max(t - 11, 0):t + 1].mean(),
             s[max(t - 23, 0):t + 1].max() - s[max(t - 23, 0):t + 1].min(),
             weather[t, 0] - weather[max(t - 24, 0), 0]]
    return np.column_stack([base, np.tile(extra, (H, 1)), dev_yday, temp_fut - weather[t, 0],
                            np.full(H, weather[t, 0])])


def rows(x, steps, feats):
    X, y, sc = [], [], []
    for t in steps:
        s = G.scale(x, t)
        f = G.features(x, hist.weather, hist.weather_pred, hist.cal, t, H)
        xs = x[:, 0] / s[0]
        extra = G.total_extra(xs, hist.weather[:, 0], t, H) if feats == 1 else \
            total_extra2(xs, hist.weather, hist.weather_pred, t, H)
        X.append(np.hstack([f, extra]).astype(np.float32))
        y.append(x[t + 1:t + H + 1, 0] / s[0])
        sc.append(np.full(H, s[0]))
    return np.concatenate(X), np.concatenate(y).astype(np.float32), np.concatenate(sc)


def pseudo(buildings, n, rng):
    out = []
    for _ in range(n):
        k = rng.integers(4, min(8, len(buildings)) + 1)
        out.append(LOAD[:, rng.choice(buildings, k, replace=False)].sum(1, keepdims=True))
    return out


def dataset(series, steps, feats):
    parts = [rows(s, steps, feats) for s in series]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def acc(pred, actual):
    return 1 - np.abs(pred - actual).mean() / np.abs(actual).mean()


BASE = {'objective': 'l1', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
        'verbose': -1, 'num_threads': THREADS, 'seed': 0}
GRID = [dict(BASE, learning_rate=lr, num_leaves=nl, min_data_in_leaf=md, feature_fraction=ff)
        for lr, nl, md, ff in [(0.05, 63, 200, 0.8), (0.03, 127, 500, 0.7), (0.05, 31, 100, 0.9),
                               (0.03, 255, 1000, 0.6), (0.08, 63, 300, 0.7), (0.03, 63, 2000, 0.8),
                               (0.05, 127, 200, 0.5), (0.02, 255, 500, 0.8)]]
steps2 = list(range(168 * 4, T - H - 1, 2))
N_PSEUDO = 45
steps_tune = list(range(168 * 4, T - H - 1, 6))

results = []
csv = OUT / 'load_lab2.csv'
for district, target in DISTRICTS.items():
    others = {d: bs for d, bs in DISTRICTS.items() if d != district}
    pool = [b for bs in others.values() for b in bs]
    actual = LOAD[:, target].sum(1, keepdims=True)
    A = np.stack([actual[t + 1:t + H + 1, 0] for t in eval_steps])
    ev = {f: rows(actual, eval_steps, f) for f in (1, 2)}

    def score(name, models, feats, note=''):
        Xe, _, se = ev[feats]
        p = np.mean([np.maximum(m.predict(Xe), 0) for m in models], axis=0) * se
        p = p.reshape(len(eval_steps), H)
        r = {'district': district, 'variant': name, 'total_acc': acc(p, A), 'note': note}
        for h in [1, 3, 6, 12, 24]:
            r[f'h{h}'] = acc(p[:, h - 1], A[:, h - 1])
        results.append(r)
        pd.DataFrame(results).to_csv(csv, index=False)
        print(f"{district} {name}: total accuracy {r['total_acc']:.3f} {note}", flush=True)

    start = time.time()
    if 'A' in VARIANTS:
        score('A_direct_x', [lgb.Booster(model_file=str(OUT / f'{district}_total_direct_x.txt'))], 1)
    data_B = None
    if 'B' in VARIANTS:
        data_B = dataset(pseudo(pool, N_PSEUDO, np.random.default_rng(0)), steps2, 1)
        m = lgb.train(BASE, lgb.Dataset(*data_B), num_boost_round=600)
        m.save_model(str(OUT / f'{district}_total_B.txt'))
        score('B_more_data', [m], 1, f'{len(data_B[1]):,} rows, {(time.time() - start) / 60:.0f} min')
    data_B = None
    data_C = None
    if any(v in VARIANTS for v in 'CDE'):
        data_C = dataset(pseudo(pool, N_PSEUDO, np.random.default_rng(0)), steps2, 2)
    if 'C' in VARIANTS:
        m = lgb.train(BASE, lgb.Dataset(*data_C), num_boost_round=600)
        m.save_model(str(OUT / f'{district}_total_C.txt'))
        score('C_more_features', [m], 2, f'{(time.time() - start) / 60:.0f} min')
    if 'D' in VARIANTS:
        ms = [m] if 'C' in VARIANTS else []
        for seed in range(1, 3 if 'C' in VARIANTS else 4):
            Xd, yd = dataset(pseudo(pool, N_PSEUDO, np.random.default_rng(seed)), steps2, 2)
            ms.append(lgb.train(dict(BASE, seed=seed), lgb.Dataset(Xd, yd), num_boost_round=600))
        for i, mm in enumerate(ms):
            mm.save_model(str(OUT / f'{district}_total_D{i}.txt'))
        score('D_3seed_ensemble', ms, 2, f'{(time.time() - start) / 60:.0f} min')
    if 'E' in VARIANTS:
        (d1, b1), (d2, b2) = others.items()
        folds = [(dataset(pseudo(b1, 30, np.random.default_rng(5)), steps_tune, 2),
                  dataset(pseudo(b2, 15, np.random.default_rng(6)), steps_tune, 2)),
                 (dataset(pseudo(b2, 30, np.random.default_rng(7)), steps_tune, 2),
                  dataset(pseudo(b1, 15, np.random.default_rng(8)), steps_tune, 2))]
        best, best_err, best_iter = None, np.inf, 600
        for p in GRID:
            errs, iters = [], []
            for (Xf, yf), (Xv, yv) in folds:
                mm = lgb.train(p, lgb.Dataset(Xf, yf), num_boost_round=3000, valid_sets=[lgb.Dataset(Xv, yv)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
                errs.append(np.abs(np.maximum(mm.predict(Xv, num_iteration=mm.best_iteration), 0) - yv).mean())
                iters.append(mm.best_iteration)
            if np.mean(errs) < best_err:
                best, best_err, best_iter = p, np.mean(errs), int(np.mean(iters) * 1.2) + 10
        m = lgb.train(best, lgb.Dataset(*data_C), num_boost_round=best_iter)
        m.save_model(str(OUT / f'{district}_total_E.txt'))
        score('E_tuned', [m], 2, f"lr {best['learning_rate']} leaves {best['num_leaves']} min_data "
              f"{best['min_data_in_leaf']} ff {best['feature_fraction']} rounds {best_iter}, "
              f'{(time.time() - start) / 60:.0f} min')

pd.set_option('display.width', 200)
print(pd.DataFrame(results).round(3).to_string(index=False))
