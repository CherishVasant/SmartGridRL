"""Value of forecast accuracy (a diagnostic, NOT a deployable controller): which forecast errors cost score?

Runs the fixed MPC of P3_GBMTUNED_MPC_g3 (tuned LightGBM forecasts, horizon 24, carbon weight 1, ramp weight 3) on
the train and val districts with parts of its forecasts replaced by the TRUE future, and scores it like
tools/phase3_eval.py. Compare with P3_GBMTUNED_MPC_g3 (train S 0.784, val S 0.779): the difference is what a perfect
forecast of that part would be worth, i.e. where forecasting effort pays off.
Modes:
  perfect_load    true per-building load for all hours, LightGBM solar
  perfect_solar   true per-building solar, LightGBM load
  perfect_total   LightGBM per-building load rescaled so the district total is the true total (shape of the total
                  right, split between buildings forecast)
  perfect_load_h6 true load for the next 6 hours only, LightGBM after that
Usage (from the project folder): python tools/forecast_value.py <mode> [districts=train,val]
Output: results/forecast/forecast_value.csv (one row per mode and district)
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase3_eval as p3   # noqa: E402  (loads the core)

ns, np, pd, RESULTS_DIR, ck = p3.ns, p3.np, p3.pd, p3.RESULTS_DIR, p3.ck
G = ns['GBMForecaster']
MODE = sys.argv[1]
TARGETS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['train', 'val']
BUILDINGS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
CFG = {'horizon': 24, 'w_carbon': 1.0, 'w_grid': 3.0, 'forecaster': 'gbm_tuned'}

O = np.load(RESULTS_DIR / 'forecast' / 'phase_all_observations.npy')
hist = G(17, 24, None, max_steps=len(O) + 1)
for t in range(len(O)):
    hist.update(t, O[t])


def make(B, d):
    ctrl = ns['make_mpc_controller'](CFG, B, d)
    fc, H = ctrl.fc, CFG['horizon']
    true_load, true_solar = hist.load[:, BUILDINGS[d]], hist.solar[:, BUILDINGS[d]]

    def future(x, t):
        idx = np.minimum(np.arange(t + 1, t + H + 1), len(O) - 1)
        return x[idx]

    def forecast(t):
        _, ci, price = ns['CausalForecaster'].forecast(fc, t)
        out = {}
        for name, x in [('load', fc.load), ('solar', fc.solar)]:
            X = fc.features(x, fc.weather, fc.weather_pred, fc.cal, t, H, fc.version)
            out[name] = np.maximum(fc.models[name].predict(X).reshape(H, B) * fc.scale(x, t), 0.0)
        if MODE == 'perfect_load':
            out['load'] = future(true_load, t)
        elif MODE == 'perfect_solar':
            out['solar'] = future(true_solar, t)
        elif MODE == 'perfect_total':
            total, parts = future(true_load, t).sum(1), out['load'].sum(1)
            out['load'] = out['load'] * (total / np.maximum(parts, 1e-6))[:, None]
        elif MODE == 'perfect_load_h6':
            out['load'][:6] = future(true_load, t)[:6]
        else:
            raise ValueError(MODE)
        return out['load'] - out['solar'], ci, price

    fc.forecast = forecast
    return ctrl.act


rows = []
for d in TARGETS:
    start = time.time()
    trace = p3.run_year(p3.DISTRICTS[d], make, d)
    k = ck.challenge_kpis(trace, p3.no_battery(d))
    rows.append({'mode': MODE, 'district': d, **{m: k[m] for m in ['C', 'G', 'D', 'phase1', 'phase3']}})
    print(f"{MODE} {d}: C {k['C']:.3f} G {k['G']:.3f} D {k['D']:.3f} S {k['phase3']:.3f} "
          f"({time.time() - start:.0f} s)", flush=True)
    log = RESULTS_DIR / 'forecast' / 'forecast_value.csv'
    pd.DataFrame(rows[-1:]).to_csv(log, mode='a', header=not log.exists(), index=False)
