"""Load-forecasting lab, round 3: the DISTRICT TOTAL over the NEXT 1-6 HOURS.

tools/forecast_value.py showed that this is where forecast accuracy pays: with the true district-total load the MPC's
S drops from 0.784 / 0.779 (train / val) to 0.649 / 0.648, and with the true load for only the next 6 hours to
0.659 / 0.651, while a perfect solar forecast is worth almost nothing. Same protocol as load_lab.py (leave-district-
out; training series = pseudo-districts of the OTHER districts' buildings; scored on the target district's real
total, every 6 h from hour 168), but scored on horizons 1-6 only. Variants:
  sum_gbm     current controller: tuned per-building LightGBM, summed                                   (reference)
  direct_x    direct total model of load_lab.py (24 h model)                                             (reference)
  S1_short    LightGBM for horizons 1-6 only, dense recent history (every lag of the last 24 h, diffs, short means,
              lags aligned to the target hour, 7 / 28-day same-hour means, weather now and forecast)
  S2_per_h    S1 features, one model per horizon (direct strategy)
  S3_x_corr   direct_x + online error correction: + a_h * (last observed 1-hour-ahead error) + b_h * (the one before)
  S4_s1_corr  S1 + the same error correction
The correction coefficients are fitted out-of-sample between the two other districts (models trained on one
district's pseudo-districts, errors of the other's), never on the target district.
Usage (from the project folder): python tools/load_lab3.py [threads=4]
Outputs: results/forecast/load_lab/<district>_short_S1.txt, short_S2_h<h>.txt, load_lab3.csv
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
HS = 6
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

O = np.load(FC / 'phase_all_observations.npy')
T = len(O)
hist = G(17, 24, None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
LOAD, W, WP, CAL = hist.load[:T], hist.weather[:T], hist.weather_pred[:T], hist.cal[:T]
eval_steps = np.arange(168, T - 25, 6)
all_steps = np.arange(168 * 4, T - HS - 1)


def short_features(x, steps):
    """(len(steps) * HS, n) features (h-major within each step) for the scaled-free total x (T,) at the given steps.
    Vectorised over steps; uses only data up to each step t (and weather forecasts issued by t)."""
    steps = np.asarray(steps)
    sc = np.array([np.abs(x[max(t - 167, 0):t + 1]).mean() + 0.05 for t in steps])
    lag = lambda k: x[np.maximum(steps - k, 0)] / sc
    lags = np.stack([lag(k) for k in range(24)], 1)                                 # (N, 24)
    roll = lambda n: np.stack([x[np.maximum(steps - k, 0)] for k in range(n)], 1).mean(1) / sc
    common = np.column_stack([lags, lags[:, 0] - lags[:, 1], lags[:, 0] - lag(24), lags[:, 0] - lag(168),
                              roll(3), roll(6), roll(12), roll(24), lags.std(1), W[steps], CAL[steps, 0]])
    rows = []
    for h in range(1, HS + 1):
        u = steps + h                                                              # target hour
        same7 = np.stack([x[np.minimum(u - 24 * d, steps)] for d in range(1, 8)], 1).mean(1) / sc
        same28 = np.stack([x[np.minimum(u - 24 * d, steps)] for d in range(1, 29)], 1).mean(1) / sc
        tgt = np.column_stack([np.full(len(steps), h), (CAL[steps, 2] - 1 + h) % 24 + 1,
                               CAL[np.maximum(u - 168, 0), 1], x[np.minimum(u - 24, steps)] / sc,
                               x[np.minimum(u - 48, steps)] / sc, x[u - 168] / sc, same7, same28,
                               WP[np.maximum(u - 6, 0), 0, 0], WP[np.maximum(u - 6, 0), 3, 0]])
        rows.append(np.column_stack([common, tgt]))
    X = np.stack(rows, 1).reshape(len(steps) * HS, -1).astype(np.float32)          # step-major, then h
    return X, sc


def short_targets(x, steps, sc):
    return (np.stack([x[np.asarray(steps) + h] for h in range(1, HS + 1)], 1) / sc[:, None]).astype(np.float32)


def pseudo(buildings, n, rng):
    out = []
    for _ in range(n):
        k = rng.integers(4, min(8, len(buildings)) + 1)
        out.append(LOAD[:, rng.choice(buildings, k, replace=False)].sum(1))
    return out


def dataset(series, steps):
    Xs, ys = [], []
    for s in series:
        X, sc = short_features(s, steps)
        Xs.append(X); ys.append(short_targets(s, steps, sc).ravel())
    return np.concatenate(Xs), np.concatenate(ys)


PARAMS = {'objective': 'l1', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200,
          'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
          'verbose': -1, 'num_threads': THREADS, 'seed': 0}
ROUNDS = 600


def predict_s1(model, x, steps):
    X, sc = short_features(x, steps)
    return np.maximum(model.predict(X), 0).reshape(len(steps), HS) * sc[:, None]


def predict_s2(models, x, steps):
    X, sc = short_features(x, steps)
    X = X.reshape(len(steps), HS, -1)
    return np.stack([np.maximum(models[h].predict(X[:, h]), 0) for h in range(HS)], 1) * sc[:, None]


def predict_x(model, x, steps):
    """direct_x (24 h total model of load_lab.py), first HS horizons."""
    xx = x[:, None]
    out = []
    for t in steps:
        s = G.scale(xx, t)
        f = np.hstack([G.features(xx, hist.weather, hist.weather_pred, hist.cal, t, 24),
                       G.total_extra(xx[:, 0] / s[0], hist.weather[:, 0], t, 24)])
        out.append(np.maximum(model.predict(f), 0)[:HS] * s[0])
    return np.stack(out)


def residuals(pred_all, x, steps):
    """Last two observed 1-hour-ahead errors at each step (from the forecasts made one and two hours earlier)."""
    p1 = dict(zip(steps, pred_all[:, 0]))
    r = lambda t: x[t] - p1[t - 1] if (t - 1) in p1 else 0.0
    return np.array([[r(t), r(t - 1)] for t in steps])


def fit_correction(preds, actuals, res):
    """Per horizon least squares of the error on the last two residuals: coefficients (HS, 2)."""
    coef = []
    for h in range(HS):
        A = np.concatenate(res)
        e = np.concatenate([a[:, h] - p[:, h] for p, a in zip(preds, actuals)])
        coef.append(np.linalg.lstsq(A, e, rcond=None)[0])
    return np.array(coef)


def acc(p, a):
    return 1 - np.abs(p - a).mean() / np.abs(a).mean()


rows = []
csv = OUT / 'load_lab3.csv'
for district, target in DISTRICTS.items():
    start = time.time()
    others = {d: bs for d, bs in DISTRICTS.items() if d != district}
    pool = [b for bs in others.values() for b in bs]
    x_t = LOAD[:, target].sum(1)
    A = np.stack([x_t[eval_steps + h] for h in range(1, HS + 1)], 1)
    hourly = np.arange(eval_steps[0] - 2, eval_steps[-1] + 1)                     # every hour (for the residuals)
    preds = {}

    # references
    model_cur = lgb.Booster(model_file=str(FC / 'tuned' / f'{district}_load.txt'))
    xb = LOAD[:, target]
    preds['sum_gbm'] = np.stack([(np.maximum(model_cur.predict(G.features(xb, hist.weather, hist.weather_pred,
                                  hist.cal, t, 24)), 0).reshape(24, len(target)) * G.scale(xb, t)).sum(1)[:HS]
                                 for t in eval_steps])
    model_x = lgb.Booster(model_file=str(OUT / f'{district}_total_direct_x.txt'))

    # S1 / S2 on pseudo-districts of the other two districts (every 2nd hour)
    Xtr, ytr = dataset(pseudo(pool, 45, np.random.default_rng(0)), all_steps[::2])
    s1 = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=ROUNDS)
    s1.save_model(str(OUT / f'{district}_short_S1.txt'))
    Xh, yh = Xtr.reshape(-1, HS, Xtr.shape[1]), ytr.reshape(-1, HS)
    s2 = [lgb.train(PARAMS, lgb.Dataset(Xh[:, h], yh[:, h]), num_boost_round=ROUNDS) for h in range(HS)]
    for h, m in enumerate(s2):
        m.save_model(str(OUT / f'{district}_short_S2_h{h + 1}.txt'))
    del Xtr, ytr, Xh, yh
    print(f'  {district}: S1 / S2 trained ({(time.time() - start) / 60:.0f} min)', flush=True)

    # error-correction coefficients, never using the target district:
    #   s1: out of sample between the two other districts (fold model fitted on one, residuals of the other)
    #   x:  direct_x's own (leave-target-out) model on pseudo-districts of the other districts (in-sample residuals,
    #       no retraining of the 24 h model)
    (d1, b1), (d2, b2) = others.items()
    hrs = np.arange(168 * 4, T - 25)
    corr = {}
    P, Aa, R = [], [], []
    for fit_b, val_b in [(b1, b2), (b2, b1)]:
        Xf, yf = dataset(pseudo(fit_b, 25, np.random.default_rng(3)), all_steps[::3])
        m = lgb.train(PARAMS, lgb.Dataset(Xf, yf), num_boost_round=ROUNDS)
        del Xf, yf
        for s in pseudo(val_b, 4, np.random.default_rng(4)):
            p = predict_s1(m, s, hrs)
            P.append(p); Aa.append(np.stack([s[hrs + h] for h in range(1, HS + 1)], 1)); R.append(residuals(p, s, hrs))
    corr['s1'] = fit_correction(P, Aa, R)
    P, Aa, R = [], [], []
    for s in pseudo(pool, 6, np.random.default_rng(4)):
        p = predict_x(model_x, s, hrs)
        P.append(p); Aa.append(np.stack([s[hrs + h] for h in range(1, HS + 1)], 1)); R.append(residuals(p, s, hrs))
    corr['x'] = fit_correction(P, Aa, R)
    print(f'  {district}: corrections fitted ({(time.time() - start) / 60:.0f} min)', flush=True)

    # target district (hourly forecasts so the residuals exist, scored at the evaluation hours)
    sel = np.isin(hourly, eval_steps)
    px, ps1 = predict_x(model_x, x_t, hourly), predict_s1(s1, x_t, hourly)
    preds['direct_x'], preds['S1_short'] = px[sel], ps1[sel]
    preds['S2_per_h'] = predict_s2(s2, x_t, eval_steps)
    preds['S3_x_corr'] = np.maximum(px + residuals(px, x_t, hourly) @ corr['x'].T, 0)[sel]
    preds['S4_s1_corr'] = np.maximum(ps1 + residuals(ps1, x_t, hourly) @ corr['s1'].T, 0)[sel]

    for name, p in preds.items():
        r = {'district': district, 'variant': name, 'acc_h1_6': acc(p, A)}
        for h in range(1, HS + 1):
            r[f'h{h}'] = acc(p[:, h - 1], A[:, h - 1])
        if name.endswith('corr'):
            r['note'] = 'a,b per h: ' + ' '.join(f'{a:.2f},{b:.2f}' for a, b in corr['x' if 'x_' in name else 's1'])
        rows.append(r)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f'{district} done in {(time.time() - start) / 60:.0f} min', flush=True)
    print(pd.DataFrame([r for r in rows if r['district'] == district]).round(3).drop(columns='note', errors='ignore')
          .to_string(index=False), flush=True)
