"""Nested, leave-district-out hyperparameter tuning of the LightGBM load / PV forecasters (Optuna).

For a target district (train = buildings 1-5, val = 6-10, test = 11-17), the model may only learn from the buildings of
the two OTHER districts. Hyperparameters are chosen by inner cross-validation between those two (fit on one, validate
on the other, both ways; early stopping on the inner validation fold). The target district is used once, at the end,
to compare the tuned model with the current one (results/forecast/<district>_<series>.txt) on the same hours. Nothing
about the target buildings influences the choice of hyperparameters.
Outputs: results/forecast/tuned/<district>_<series>.txt, best_params.json, tuning_summary.csv.
Usage (from the project folder): python tools/tune_forecaster.py [n_trials=30] [district ...]
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import os
import sys
import time

import lightgbm as lgb
import optuna

ns = {'__name__': '__main__'}
for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', '00_core.ipynb'),
                        encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, pd, RESULTS_DIR = ns['np'], ns['pd'], ns['RESULTS_DIR']
G = ns['GBMForecaster']
OUT = RESULTS_DIR / 'forecast' / 'tuned'
OUT.mkdir(parents=True, exist_ok=True)
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 24
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TARGETS = sys.argv[2:] or list(DISTRICTS)

O = np.load(RESULTS_DIR / 'forecast' / 'phase_all_observations.npy')
T = len(O)
hist = G(17, H, None, max_steps=T + 1)
for t in range(T):
    hist.update(t, O[t])
SERIES = {'load': hist.load, 'solar': hist.solar}


def dataset(x, buildings, steps):
    """Features, scaled targets and scales (as forecast_lab.py / GBMForecaster) for buildings at decision times."""
    X, y, sc = [], [], []
    xb = x[:, buildings]
    for t in steps:
        X.append(G.features(xb, hist.weather, hist.weather_pred, hist.cal, t, H))
        s = G.scale(xb, t)
        y.append(np.array([xb[t + h] / s for h in range(1, H + 1)]).ravel())
        sc.append(np.tile(s, H))
    return np.concatenate(X), np.concatenate(y), np.concatenate(sc)


tune_steps = list(range(168, T - H - 1, 6))      # tuning: every 6th hour (faster)
fit_steps = list(range(168, T - H - 1, 2))       # final fit: every 2nd hour (as the current models)
eval_steps = list(range(168, T - H - 1, 6))      # target evaluation: same hours as forecast_lab.py
cache = {}


def data(series, buildings, steps_name):
    key = (series, tuple(buildings), steps_name)
    if key not in cache:
        cache[key] = dataset(SERIES[series], buildings, {'tune': tune_steps, 'fit': fit_steps, 'eval': eval_steps}[steps_name])
    return cache[key]


def params_of(trial):
    p = {'objective': trial.suggest_categorical('objective', ['l1', 'l2', 'huber']),
         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
         'num_leaves': trial.suggest_int('num_leaves', 15, 255, log=True),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 1000, log=True),
         'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0), 'bagging_freq': 1,
         'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 30.0, log=True),
         'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
         'verbose': -1, 'num_threads': 6, 'seed': 0}
    return p


# Resume: (district, series) pairs already tuned (model saved and summarised) are kept and skipped
rows, best_all = [], {}
if (OUT / 'tuning_summary.csv').exists() and (OUT / 'best_params.json').exists():
    rows = pd.read_csv(OUT / 'tuning_summary.csv').to_dict('records')
    best_all = json.load(open(OUT / 'best_params.json'))
done = {(r['district'], r['series']) for r in rows if (OUT / f"{r['district']}_{r['series']}.txt").exists()}
rows = [r for r in rows if (r['district'], r['series']) in done]
for district in TARGETS:
    others = [d for d in DISTRICTS if d != district]
    for series in SERIES:
        if (district, series) in done:
            print(f'{district} {series}: already tuned - skipped', flush=True)
            continue
        start = time.time()
        folds = [(others[0], others[1]), (others[1], others[0])]     # (fit on, validate on)

        def objective(trial):
            p = params_of(trial)
            maes, iters = [], []
            for fit_d, val_d in folds:
                Xf, yf, _ = data(series, DISTRICTS[fit_d], 'tune')
                Xv, yv, sv = data(series, DISTRICTS[val_d], 'tune')
                model = lgb.train(p, lgb.Dataset(Xf, yf), num_boost_round=2000,
                                  valid_sets=[lgb.Dataset(Xv, yv)],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
                pred = np.maximum(model.predict(Xv, num_iteration=model.best_iteration), 0)
                maes.append(np.abs((pred - yv) * sv).mean())          # kWh
                iters.append(model.best_iteration)
            trial.set_user_attr('best_iter', int(np.mean(iters)))
            return float(np.mean(maes))

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=N_TRIALS)
        best = dict(params_of(study.best_trial), verbose=-1, num_threads=6, seed=0)
        rounds = int(study.best_trial.user_attrs['best_iter'] * 1.15) + 10     # a bit more data in the final fit

        # final fit on both other districts, then the one-off comparison on the target district
        Xtr = np.concatenate([data(series, DISTRICTS[d], 'fit')[0] for d in others])
        ytr = np.concatenate([data(series, DISTRICTS[d], 'fit')[1] for d in others])
        tuned = lgb.train(best, lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
        tuned.save_model(str(OUT / f'{district}_{series}.txt'))
        Xe, ye, se = data(series, DISTRICTS[district], 'eval')
        current = lgb.Booster(model_file=str(RESULTS_DIR / 'forecast' / f'{district}_{series}.txt'))
        mae_tuned = np.abs((np.maximum(tuned.predict(Xe), 0) - ye) * se).mean()
        mae_current = np.abs((np.maximum(current.predict(Xe), 0) - ye) * se).mean()
        actual = np.abs(ye * se).mean()
        best_all[f'{district}_{series}'] = dict(best, num_boost_round=rounds, inner_cv_mae=study.best_value)
        rows.append({'district': district, 'series': series, 'inner_cv_mae_kwh': study.best_value,
                     'target_mae_current': mae_current, 'target_mae_tuned': mae_tuned,
                     'accuracy_current': 1 - mae_current / actual, 'accuracy_tuned': 1 - mae_tuned / actual,
                     'improvement_pct': 100 * (1 - mae_tuned / mae_current), 'rounds': rounds,
                     'minutes': (time.time() - start) / 60})
        print(f"{district} {series}: target MAE current {mae_current:.4f} -> tuned {mae_tuned:.4f} kWh "
              f"({rows[-1]['improvement_pct']:+.1f}%), accuracy {rows[-1]['accuracy_current']:.1%} -> "
              f"{rows[-1]['accuracy_tuned']:.1%} | {N_TRIALS} trials, {rows[-1]['minutes']:.0f} min", flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'tuning_summary.csv', index=False)
        with open(OUT / 'best_params.json', 'w') as f:
            json.dump(best_all, f, indent=2)
print('done ->', OUT, flush=True)
