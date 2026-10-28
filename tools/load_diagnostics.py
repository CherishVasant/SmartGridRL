"""Where the load-forecast error is: per building vs district total, by horizon, and how predictable the load is at all.

For each district (models trained leave-district-out, as used by the MPC) on the forecast_lab.py evaluation hours:
  * accuracy = 1 - MAE / mean|actual|, per building and for the district total (the KPIs R and L use the total),
  * by horizon h = 1..24, for the tuned LightGBM, the current LightGBM and the causal seasonal forecast,
  * persistence baselines (last hour, same hour yesterday, same hour last week),
  * noise share: fraction of the load's hourly variance that is not explained by its 7-day same-hour mean.
Usage (from the project folder): python tools/load_diagnostics.py   -> results/forecast/load_diagnostics.csv
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import os

import lightgbm as lgb

ns = {'__name__': '__main__'}
for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', '00_core.ipynb'),
                        encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
G = ns['GBMForecaster']
FC = RESULTS_DIR / 'forecast'
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 24

O = np.load(FC / 'phase_all_observations.npy')
T = len(O)
hist = G(17, H, None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
load = hist.load[:T]
steps = list(range(168, T - H - 1, 6))


def acc(pred, actual):
    return 1 - np.abs(pred - actual).mean() / np.abs(actual).mean()


rows = []
for d, b in DISTRICTS.items():
    x = load[:, b]
    actual = np.stack([x[t + 1:t + H + 1] for t in steps])                        # (N, H, B)
    preds = {}
    for name, path in [('gbm_tuned', FC / 'tuned' / f'{d}_load.txt'), ('gbm', FC / f'{d}_load.txt')]:
        model = lgb.Booster(model_file=str(path))
        p = []
        for t in steps:
            X = G.features(x, hist.weather, hist.weather_pred, hist.cal, t, H)
            p.append(np.maximum(model.predict(X), 0).reshape(H, len(b)) * G.scale(x, t))
        preds[name] = np.stack(p)
    seas = []
    for t in steps:
        err = x[t] - hist._seasonal(x, t, 0)
        seas.append(np.maximum([hist._seasonal(x, t, h) + err * 0.8 ** h for h in range(1, H + 1)], 0))
    preds['seasonal'] = np.stack(seas)
    preds['last_hour'] = np.stack([np.repeat(x[t][None], H, 0) for t in steps])
    preds['yesterday'] = np.stack([x[[min(t + h - 24, t) for h in range(1, H + 1)]] for t in steps])
    preds['last_week'] = np.stack([x[[t + h - 168 for h in range(1, H + 1)]] for t in steps])
    for name, p in preds.items():
        row = {'district': d, 'forecast': name, 'building_acc': acc(p, actual),
               'district_total_acc': acc(p.sum(-1), actual.sum(-1))}
        for h in [1, 2, 3, 6, 12, 24]:
            row[f'bld_h{h}'] = acc(p[:, h - 1], actual[:, h - 1])
            row[f'tot_h{h}'] = acc(p[:, h - 1].sum(-1), actual[:, h - 1].sum(-1))
        rows.append(row)
    # noise share: variance of (load - 7-day same-hour mean) / variance of load, per building and for the total
    for label, series in [('building', x), ('district_total', x.sum(1, keepdims=True))]:
        dev = np.stack([series[t] - np.mean([series[t - 24 * k] for k in range(1, 8)], axis=0) for t in range(168, T)])
        rows.append({'district': d, 'forecast': f'noise_share_{label}',
                     'building_acc': float((dev.var(0) / series[168:].var(0)).mean())})
    print(d, 'done', flush=True)

df = pd.DataFrame(rows)
df.to_csv(FC / 'load_diagnostics.csv', index=False)
pd.set_option('display.width', 250)
print(df.round(3).to_string(index=False))
