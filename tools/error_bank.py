"""Forecast-error bank for the scenario MPC (ScenarioMPC in the core).

For each target district d, collects the errors of the controller's own forecaster (tuned per-building LightGBM,
'gbm_tuned', district-total net load = load - solar, horizons 1..36) on the OTHER two districts, each forecast by its
own leave-district-out models, so every error is out of sample and nothing comes from district d. Errors are divided
by the district's past-week mean total load, so they transfer between districts of different size.
Output: results/forecast/error_bank/<district>.npz with errors (N, 36) and the hour of day (1..24) of each forecast.
Usage (from the project folder): python tools/error_bank.py
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import json
import os

ns = {'__name__': '__main__'}
for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', '00_core.ipynb'),
                        encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        exec(''.join(c['source']), ns)
np, RESULTS_DIR = ns['np'], ns['RESULTS_DIR']
G = ns['GBMForecaster']
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
H = 36
OUT = RESULTS_DIR / 'forecast' / 'error_bank'
OUT.mkdir(parents=True, exist_ok=True)

O = np.load(RESULTS_DIR / 'forecast' / 'phase_all_observations.npy')
T = len(O)
full = G(17, 24, None, max_steps=T + 1)
for t in range(T):
    full.update(t, O[t])


def district_errors(d):
    """Out-of-sample relative errors of district d's own controller forecaster on district d."""
    b = DISTRICTS[d]
    fc = ns['make_mpc_controller']({'horizon': H, 'w_carbon': 1.0, 'w_grid': 3.0, 'forecaster': 'gbm_tuned'},
                                   len(b), d).fc
    fc.load[:T], fc.solar[:T] = full.load[:T, b], full.solar[:T, b]
    fc.weather[:T], fc.weather_pred[:T], fc.cal[:T] = full.weather[:T], full.weather_pred[:T], full.cal[:T]
    fc.ci[:T], fc.price[:T] = full.ci[:T], full.price[:T]
    net = (full.load[:T, b] - full.solar[:T, b]).sum(1)
    errs, hours = [], []
    for t in range(168, T - H - 1):
        f = fc.forecast(t)[0].sum(1)
        scale = full.load[max(t - 167, 0):t + 1, b].sum(1).mean() + 0.05
        errs.append((net[t + 1:t + H + 1] - f) / scale)
        hours.append(full.cal[t, 2])
    return np.array(errs, dtype=np.float32), np.array(hours, dtype=np.int8)


per = {d: district_errors(d) for d in DISTRICTS}
for d in DISTRICTS:
    others = [o for o in DISTRICTS if o != d]
    np.savez(OUT / f'{d}.npz', errors=np.concatenate([per[o][0] for o in others]),
             hours=np.concatenate([per[o][1] for o in others]))
    e = np.concatenate([per[o][0] for o in others])
    print(f'{d}: {len(e)} error vectors from {others}; mean |error| h1 {np.abs(e[:, 0]).mean():.3f}, '
          f'h6 {np.abs(e[:, 5]).mean():.3f}, h24 {np.abs(e[:, 23]).mean():.3f} (x past-week mean load)', flush=True)
