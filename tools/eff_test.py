"""Battery-efficiency fix for the MPC (Round 33): fixed MPC of P3_GBMTUNED_MPC_g3 (tuned LightGBM forecasts, horizon 24,
carbon weight 1, ramp weight 3) on one district, with the action conversion and the LP's efficiency varied.
Also reports how closely the batteries reach the MPC's target state of charge.
Variants: base (constant efficiency, as before) | exact (CityLearn's power-dependent efficiency curve when converting
the target SoC into an action) | exact_eta<x> (exact + LP one-way efficiency x, e.g. exact_eta0.93)
Usage (from the project folder): python tools/eff_test.py <variant> [district=train]
Output: results/phase3/eff_test.csv (one row per variant and district)
"""
import torch   # noqa: F401  (before pandas: WinError 1114)
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase3_eval as p3   # noqa: E402  (loads the core)

ns, np, pd, RESULTS_DIR, ck = p3.ns, p3.np, p3.pd, p3.RESULTS_DIR, p3.ck
VARIANT = sys.argv[1]
DISTRICT = sys.argv[2] if len(sys.argv) > 2 else 'train'
cfg = {'horizon': 24, 'w_carbon': 1.0, 'w_grid': 3.0, 'forecaster': 'gbm_tuned',
       'exact_eff': VARIANT.startswith('exact'), 'eta': float(VARIANT.split('eta')[1]) if 'eta' in VARIANT else None}
miss = []


def make(B, d):
    ctrl = ns['make_mpc_controller'](cfg, B, d)
    s, k, CAP = ns['CENTRAL_SHARED_OBS'], ns['CENTRAL_OBS_PER_BUILDING'], ns['BatteryMPC'].CAP
    state = {'target': None}

    def policy(t, o, soc):
        if state['target'] is not None:
            miss.append(np.abs(np.array(soc) - state['target'] / CAP))
        ctrl.observe(t, o, soc)
        m = ctrl.mpc
        state['target'] = m.plan(ctrl.base_fc, ctrl.ci_fc, ctrl.price_fc, ctrl.soc0, ctrl.E0, ctrl.scales,
                                 peak_ref=ctrl.month_peak, mean_load=ctrl.month_sum / ctrl.month_n)
        return ctrl.decide()
    return policy


start = time.time()
trace = p3.run_year(p3.DISTRICTS[DISTRICT], make, DISTRICT)
k = ck.challenge_kpis(trace, p3.no_battery(DISTRICT))
row = {'variant': VARIANT, 'district': DISTRICT, **{m: k[m] for m in ['C', 'G', 'R', '1-L', 'D', 'phase1', 'phase3']},
       'soc_miss': float(np.mean(miss))}
print(f"{VARIANT} {DISTRICT}: C {k['C']:.3f} G {k['G']:.3f} R {k['R']:.3f} 1-L {k['1-L']:.3f} S {k['phase3']:.3f} "
      f"| mean SoC miss {row['soc_miss']:.4f} ({time.time() - start:.0f} s)", flush=True)
log = RESULTS_DIR / 'phase3' / 'eff_test.csv'
pd.DataFrame([row]).to_csv(log, mode='a', header=not log.exists(), index=False)
