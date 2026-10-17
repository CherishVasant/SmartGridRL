# %% [markdown]
# # 00 Core: shared code for phase 3 (multi-KPI adaptive reward)
#
# Every method notebook (`B*` baselines, `M*` new methods) and the comparison notebook start with
# `%run ./00_core.ipynb`. This notebook only defines things; it trains nothing (except the RBC reference,
# which runs once, takes about a minute and is cached).
#
# **Contents**
# 1. Imports and configuration (training length, adaptive-weight settings, method definitions)
# 2. Per-step terms: cost, carbon, ramping, peak increase, battery wear (per building and district)
# 3. RBC reference run: used to normalise reward terms and as the KPI baseline (ratio = 1.0)
# 4. KPIs: the CityLearn-style set, reported as ratios to RBC
# 5. Rewards: cost-only, static, dynamic (linear or top-25% φ), multi-KPI (fixed or adaptive learned weights)
# 6. Environments: central agent (one policy, all 5 batteries) and per-building agents (one shared policy)
# 7. Resumable training: checkpoints every `CHECKPOINT_EVERY` env steps; evaluation; `run_method`
#
# **Results layout**: `results/<METHOD>/seed_<seed>/` holds `checkpoint.zip` + `checkpoint.json` (while training),
# `model.zip`, `train_log.csv`, `eval_trace.csv` and `summary.json` (a seed is finished when `summary.json` exists).

# %%
# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

warnings.filterwarnings('ignore')

# One thread per run: small MLPs train as fast on 1 thread, and several notebooks can run side by side
torch.set_num_threads(1)

# %% [markdown]
# ## 1. Configuration

# %%
# This notebook lives in <repo>/02_multi_kpi_adaptive_reward; found by name so any Jupyter start folder works
PHASE_DIR_NAME = '02_multi_kpi_adaptive_reward'
_cwd = Path.cwd().resolve()
PHASE_DIR = next((p for p in [_cwd, *_cwd.parents] if p.name == PHASE_DIR_NAME), _cwd / PHASE_DIR_NAME)

# Smoke test (set env var SMARTGRID_SMOKE_TEST=1): tiny runs written to a separate folder, for checking the code only
SMOKE_TEST = os.environ.get('SMARTGRID_SMOKE_TEST') == '1'
RESULTS_DIR = PHASE_DIR / ('results_smoke_test' if SMOKE_TEST else 'results')
PLOTS_DIR = RESULTS_DIR / 'plots'
for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset & environment setup (same as phase 2)
DATASET_NAME = 'citylearn_challenge_2022_phase_1'

# Training length, counted in ENVIRONMENT steps (1 episode = 1 year = 8,759 steps), the same for every method.
# A per-building run collects 5 samples per env step (one per building) but uses the same env-step budget.
TOTAL_ENV_STEPS = 3_000 if SMOKE_TEST else 200_000
CHECKPOINT_EVERY = 1_000 if SMOKE_TEST else 20_000   # env steps between resumable checkpoints
EVAL_STEPS = 400 if SMOKE_TEST else None             # None = evaluate on the full year

# PPO hyperparameters (identical to phase 2)
PPO_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 128,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'verbose': 0,
}
POLICY_KWARGS = {'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}

# SAC (M12): off-policy, one gradient step per env step; replay buffer capped so one run stays around 1 GB
SAC_PARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 200_000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'learning_starts': 1_000,
    'verbose': 0,
}
SAC_POLICY_KWARGS = {'net_arch': [64, 64]}

# Carbon weights from phase 2: LAMBDA_MATCHED = LAMBDA_MAIN * mean(I_hat), the static weight with Dynamic's average
LAMBDA_MAIN = 1.0
LAMBDA_MATCHED = 0.41

# Multi-KPI reward: every term is divided by its RBC per-step average, so all terms have comparable scale
W_BATT = 0.1                     # fixed small weight on battery wear (|delta SoC|)

# Adaptive (learned) KPI weights, in the style of Dynamic Weight Average (multi-task learning).
# Once per window, for each KPI k: gap_k = log((KPI_k / KPI_k^RBC over that window) / target_k)  (> 0 = behind target).
# New weights = ADAPT_TOTAL * (ADAPT_FLOOR / K + (1 - ADAPT_FLOOR) * softmax(gap / ADAPT_TEMPERATURE)), blended into the
# old ones with ADAPT_SMOOTHING. The log keeps one very bad KPI from taking all the weight, and the floor keeps every
# KPI (carbon included) at >= ADAPT_TOTAL * ADAPT_FLOOR / K = 0.3.
# The weights always sum to ADAPT_TOTAL (= 3, the same total as M2's fixed weights), so they cannot all run away;
# the weight moves to whichever KPI is furthest behind its target. (A first version with Lagrangian-style additive
# updates pushed every weight to its cap within two episodes, because early policies trail RBC on every KPI.)
ADAPT_KPIS = ['carbon', 'ramp', 'peak']
ADAPT_TARGETS = {'carbon': 0.90, 'ramp': 1.0, 'peak': 1.0}   # carbon: aim 10% below RBC; grid KPIs: no worse than RBC
ADAPT_WINDOW = 168               # env steps (1 week) between weight updates
ADAPT_TOTAL = float(len(ADAPT_KPIS))
ADAPT_TEMPERATURE = 1.0
ADAPT_FLOOR = 0.3
ADAPT_SMOOTHING = 0.2            # mu <- (1 - s) * mu + s * new weights
MU_INIT = 1.0

# Methods. agent: 'central' (one policy, 5 actions) or 'per_building' (one shared policy, 1 action per building)
# reward: 'cost_only' | 'static' | 'dynamic' | 'multikpi';  phi: carbon weight scaling ('none' | 'linear' | 'top25')
METHODS = {
    'B1_cost_only':        {'agent': 'central', 'reward': 'cost_only', 'label': 'Cost-Only'},
    'B3_static_matched':   {'agent': 'central', 'reward': 'static', 'lam': LAMBDA_MATCHED, 'label': 'Static-Matched'},
    'B4_dynamic':          {'agent': 'central', 'reward': 'dynamic', 'lam': LAMBDA_MAIN, 'phi': 'linear', 'label': 'Dynamic'},
    'M1_dynamic_sharp':    {'agent': 'central', 'reward': 'dynamic', 'lam': LAMBDA_MAIN, 'phi': 'top25', 'label': 'Dynamic (top-25% φ)'},
    'M2_multikpi_fixed':   {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'label': 'Multi-KPI fixed'},
    'M3_multikpi_adaptive': {'agent': 'central', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear', 'label': 'Multi-KPI adaptive'},
    'M4_per_building_dynamic': {'agent': 'per_building', 'reward': 'dynamic', 'lam': LAMBDA_MAIN, 'phi': 'linear', 'label': 'Per-building Dynamic'},
    'M5_per_building_adaptive_shared': {'agent': 'per_building', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear',
                                        'mu_per_building': False, 'label': 'Per-building adaptive (shared μ)'},
    'M6_per_building_adaptive_own': {'agent': 'per_building', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear',
                                     'mu_per_building': True, 'label': 'Per-building adaptive (own μ)'},
    # Round 1 (after Stage A): normalisation, load-squared reward, district coordination, SAC
    'M7_multikpi_adaptive_norm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear', 'normalize': True,
                                  'label': 'Multi-KPI adaptive + norm'},
    'M8_multikpi_fixed_norm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'normalize': True,
                               'label': 'Multi-KPI fixed + norm'},
    'M9_load_squared': {'agent': 'central', 'reward': 'load_squared', 'label': 'Cost + carbon + load²'},
    'M10_per_building_dynamic_district': {'agent': 'per_building', 'reward': 'dynamic', 'lam': LAMBDA_MAIN, 'phi': 'linear',
                                          'district_penalty': True, 'label': 'Per-building Dynamic + district penalty'},
    'M11_per_building_adaptive_district': {'agent': 'per_building', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear',
                                           'mu_per_building': False, 'local_grid_terms': False,
                                           'label': 'Per-building adaptive (district grid terms)'},
    'M12_sac_multikpi_fixed': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'label': 'SAC multi-KPI fixed'},
    # Round 2 (after Round 1): combine the two biggest wins, SAC (M12) and VecNormalize (M7, M8)
    'M13_sac_multikpi_fixed_norm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                    'normalize': True, 'label': 'SAC multi-KPI fixed + norm'},
    'M14_sac_multikpi_adaptive_norm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': True, 'phi': 'linear', 'algo': 'sac',
                                       'normalize': True, 'label': 'SAC multi-KPI adaptive + norm'},
    'M15_sac_load_squared_norm': {'agent': 'central', 'reward': 'load_squared', 'algo': 'sac', 'normalize': True,
                                  'label': 'SAC cost + carbon + load² + norm'},
    # Residual RL: battery action = RBC action for that hour + the policy's correction (clipped to the action bounds)
    'M16_sac_residual_rbc_norm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                  'normalize': True, 'residual': True, 'label': 'Residual SAC on RBC + norm'},
    # Round 3: double down on SAC + the fixed multi-KPI reward (M12, best overall; M13, most stable)
    'M17_sac_multikpi_obsnorm': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                 'normalize': True, 'normalize_reward': False, 'label': 'SAC multi-KPI + obs-only norm'},
    'M18_sac_multikpi_batt05': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                'w_batt': 0.5, 'label': 'SAC multi-KPI, battery wear 0.5'},
    'M19_sac_multikpi_carbon2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                 'mu_fixed': {'carbon': 2.0}, 'label': 'SAC multi-KPI, carbon weight 2'},
    'M20_sac_multikpi_counterfactual': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                        'counterfactual': True, 'label': 'SAC multi-KPI, gain over no battery'},
    # Round 4: combine and push the two Round 3 winners (M19 carbon weight 2, M20 counterfactual), all with the
    # full-day peak term (fixes the free first-hour spike that the counterfactual reward even paid for)
    'M21_sac_cf_carbon2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'mu_fixed': {'carbon': 2.0},
                           'label': 'SAC gain over no battery, carbon 2'},
    'M22_sac_cf_carbon3': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'mu_fixed': {'carbon': 3.0},
                           'label': 'SAC gain over no battery, carbon 3'},
    'M23_sac_cf_carbon2_ramp2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                 'counterfactual': True, 'full_day_peak': True, 'mu_fixed': {'carbon': 2.0, 'ramp': 2.0},
                                 'label': 'SAC gain over no battery, carbon 2, ramp 2'},
    'M24_sac_cf_peakfix': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True,
                           'label': 'SAC gain over no battery, full-day peak (M20 fixed)'},
    # Round 5: score-aligned rewards. Official district-level cost / carbon (exports earn nothing; buildings offset each
    # other), terms scaled by their no-battery per-step mean, weights shaped like the challenge score
    # (C + G + (R + 1-L) / 2) / 3, i.e. cost 1, carbon 1, ramping 0.5, peak 0.5 (proxy for 1 - load factor)
    'M25_sac_official_cf': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                            'counterfactual': True, 'full_day_peak': True, 'official_terms': True,
                            'label': 'SAC official cost/carbon terms, gain over no battery'},
    'M26_sac_score_aligned': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                              'counterfactual': True, 'full_day_peak': True, 'official_terms': True,
                              'norm_ref': 'no_battery', 'mu_fixed': {'ramp': 0.5, 'peak': 0.5},
                              'label': 'SAC score-aligned (official terms, no-battery scale, C+G+D/2)'},
    'M27_sac_score_aligned_carbon2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                      'counterfactual': True, 'full_day_peak': True, 'official_terms': True,
                                      'norm_ref': 'no_battery', 'mu_fixed': {'carbon': 2.0, 'ramp': 0.5, 'peak': 0.5},
                                      'label': 'SAC score-aligned, carbon 2'},
    'M28_sac_phase1_aligned': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'counterfactual': True, 'full_day_peak': True, 'official_terms': True,
                               'norm_ref': 'no_battery', 'mu_fixed': {'ramp': 0.2, 'peak': 0.2},
                               'label': 'SAC Phase I-aligned (cost + carbon, light grid terms)'},
    # Round 6 (fusion): residual SAC around a perfect-foresight LP plan (tools/lp_oracle.py, results/O*): action = plan's
    # action for that hour + scaled SAC correction, so RL fixes what the linear battery model gets wrong
    'M29_resid_lp_phase1': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                            'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                            'mu_fixed': {'ramp': 0.2, 'peak': 0.2}, 'residual': True, 'residual_base': 'O3_lp_phase1_eff087',
                            'residual_scale': 0.25, 'label': 'Residual SAC on Phase I LP plan (Phase I reward)'},
    'M30_resid_lp_grid': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                          'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                          'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'residual': True, 'residual_base': 'O2_lp_oracle_grid',
                          'residual_scale': 0.25, 'label': 'Residual SAC on grid LP plan (score reward)'},
    'M31_resid_lp_grid_carbon2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                  'counterfactual': True, 'full_day_peak': True, 'official_terms': True,
                                  'norm_ref': 'no_battery', 'mu_fixed': {'carbon': 2.0, 'ramp': 0.5, 'peak': 0.5},
                                  'residual': True, 'residual_base': 'O2_lp_oracle_grid', 'residual_scale': 0.25,
                                  'label': 'Residual SAC on grid LP plan (score reward, carbon 2)'},
    'M32_resid_lp_grid_full': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                               'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'residual': True, 'residual_base': 'O2_lp_oracle_grid',
                               'residual_scale': 1.0, 'label': 'Residual SAC on grid LP plan, full-range correction'},
    # Round 7: behaviour cloning of the LP plans (the policy needs only the current observation, not the plan), alone
    # (env_steps 1000 = no SAC update, since learning starts after 1,000 steps) or as the start of SAC fine-tuning
    'M33_bc_lp_grid': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                       'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                       'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                       'bc_from': 'O2_lp_oracle_grid', 'env_steps': 1000, 'label': 'Behaviour clone of grid LP plan'},
    'M34_bc_lp_grid_sac': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                           'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                           'bc_from': 'O2_lp_oracle_grid', 'sac_params': {'ent_coef': 'auto_0.05'},
                           'label': 'Behaviour clone of grid LP plan + SAC fine-tuning (score reward)'},
    'M35_bc_lp_phase1': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                         'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                         'mu_fixed': {'ramp': 0.2, 'peak': 0.2}, 'normalize': True, 'normalize_reward': False,
                         'bc_from': 'O3_lp_phase1_eff087', 'env_steps': 1000, 'label': 'Behaviour clone of Phase I LP plan'},
    'M36_bc_lp_phase1_sac': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                             'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                             'mu_fixed': {'ramp': 0.2, 'peak': 0.2}, 'normalize': True, 'normalize_reward': False,
                             'bc_from': 'O3_lp_phase1_eff087', 'sac_params': {'ent_coef': 'auto_0.05'},
                             'label': 'Behaviour clone of Phase I LP plan + SAC fine-tuning (Phase I reward)'},
    # Round 8: better imitation of the smooth grid plan (cloning O2 beat cloning O3 even on the Phase I score):
    # a bigger network, DAgger (states the clone visits, labelled with the action that steers back to the plan's SoC),
    # and a smoother plan to clone (O8, grid weight 2)
    'M37_bc_grid_big_dagger': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                               'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                               'bc_from': 'O2_lp_oracle_grid', 'bc_dagger_iters': 3, 'net_arch': [256, 256],
                               'env_steps': 1000, 'label': 'Clone of grid LP plan, 256x256 + DAgger'},
    'M38_bc_grid_big': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                        'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                        'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                        'bc_from': 'O2_lp_oracle_grid', 'net_arch': [256, 256], 'env_steps': 1000,
                        'label': 'Clone of grid LP plan, 256x256'},
    'M39_bc_grid_dagger': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                           'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                           'bc_from': 'O2_lp_oracle_grid', 'bc_dagger_iters': 3, 'env_steps': 1000,
                           'label': 'Clone of grid LP plan + DAgger'},
    'M40_bc_grid2_big_dagger': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                                'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                                'bc_from': 'O8_lp_grid2', 'bc_dagger_iters': 3, 'net_arch': [256, 256],
                                'env_steps': 1000, 'label': 'Clone of smoother LP plan (grid weight 2), 256x256 + DAgger'},
    # Round 9: gentle DAgger (30% of the SoC error per hour instead of a one-hour catch-up) with the big clone, and the big
    # clone as the start of SAC fine-tuning with the best RL reward so far (M27: score-aligned, carbon 2)
    'M41_bc_grid_big_gentle': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                               'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                               'bc_from': 'O2_lp_oracle_grid', 'bc_dagger_iters': 3, 'bc_dagger_gain': 0.3,
                               'net_arch': [256, 256], 'env_steps': 1000,
                               'label': 'Clone of grid LP plan, 256x256 + gentle DAgger'},
    'M42_bc_grid2_big_gentle': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                                'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                                'bc_from': 'O8_lp_grid2', 'bc_dagger_iters': 3, 'bc_dagger_gain': 0.3,
                                'net_arch': [256, 256], 'env_steps': 1000,
                                'label': 'Clone of smoother LP plan, 256x256 + gentle DAgger'},
    'M43_bc_grid_big_sac_c2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                               'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                               'mu_fixed': {'carbon': 2.0, 'ramp': 0.5, 'peak': 0.5}, 'normalize': True,
                               'normalize_reward': False, 'bc_from': 'O2_lp_oracle_grid', 'net_arch': [256, 256],
                               'sac_params': {'ent_coef': 'auto_0.05'},
                               'label': 'Big clone of grid LP plan + SAC fine-tuning (score reward, carbon 2)'},
    'M44_bc_grid2_big_sac_c2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                                'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                                'mu_fixed': {'carbon': 2.0, 'ramp': 0.5, 'peak': 0.5}, 'normalize': True,
                                'normalize_reward': False, 'bc_from': 'O8_lp_grid2', 'net_arch': [256, 256],
                                'sac_params': {'ent_coef': 'auto_0.05'},
                                'label': 'Big clone of smoother LP plan + SAC fine-tuning (score reward, carbon 2)'},
    # The 256x256 clone alone (M38) was worse than the 64x64 one (M33) despite a 10x lower fit error: it overfits the
    # plan's own trajectory and recovers worse from drift. So the two best Round 9 ideas also get the 64x64 actor:
    'M45_bc_grid_gentle': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                           'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                           'bc_from': 'O2_lp_oracle_grid', 'bc_dagger_iters': 3, 'bc_dagger_gain': 0.3, 'env_steps': 1000,
                           'label': 'Clone of grid LP plan + gentle DAgger'},
    'M46_bc_grid_sac_c2': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                           'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                           'mu_fixed': {'carbon': 2.0, 'ramp': 0.5, 'peak': 0.5}, 'normalize': True,
                           'normalize_reward': False, 'bc_from': 'O2_lp_oracle_grid', 'sac_params': {'ent_coef': 'auto_0.05'},
                           'label': 'Clone of grid LP plan + SAC fine-tuning (score reward, carbon 2)'},
    # Round 10: the 3-seed ensemble of M33 clones beat every single seed (Phase I 0.734 vs 0.747-0.753), so M47 trains the
    # same clone on 7 more seeds (1-7) purely as ensemble members (tools/ensemble_eval.py); it is not a separate method
    'M47_bc_grid_members': {'agent': 'central', 'reward': 'multikpi', 'adaptive': False, 'phi': 'none', 'algo': 'sac',
                            'counterfactual': True, 'full_day_peak': True, 'official_terms': True, 'norm_ref': 'no_battery',
                            'mu_fixed': {'ramp': 0.5, 'peak': 0.5}, 'normalize': True, 'normalize_reward': False,
                            'bc_from': 'O2_lp_oracle_grid', 'env_steps': 1000,
                            'label': 'Clone of grid LP plan (as M33), extra seeds for ensembles'},
}
# Round 11: double down on M41 (256x256 clone + gentle DAgger), the best single model (seed 42: Phase I 0.720,
# mean(C, G, D) 0.751, G 0.775). Each variant changes one thing.
_M41 = METHODS['M41_bc_grid_big_gentle']
METHODS.update({
    'M48_bc_big_gentle_members': dict(_M41, label='Big clone + gentle DAgger (as M41), extra seeds for ensembles'),
    'M49_bc_big_gentle_dagger6': dict(_M41, bc_dagger_iters=6, label='Big clone + gentle DAgger, 6 iterations'),
    'M50_bc_big_gentle_gain05': dict(_M41, bc_dagger_gain=0.5, label='Big clone + gentle DAgger, gain 0.5'),
    'M51_bc_deep_gentle': dict(_M41, net_arch=[256, 256, 256], label='Deep clone (3x256) + gentle DAgger'),
})
# Round 12: double down on M51 (3x256 + gentle DAgger; seed 42: Phase I 0.685, mean(C, G, D) 0.687, G 0.745)
_M51 = METHODS['M51_bc_deep_gentle']
METHODS.update({
    'M52_bc_deep_members': dict(_M51, label='Deep clone + gentle DAgger (as M51), extra seeds for ensembles'),
    'M53_bc_deeper_gentle': dict(_M51, net_arch=[256, 256, 256, 256], label='Deeper clone (4x256) + gentle DAgger'),
    'M54_bc_wide_gentle': dict(_M51, net_arch=[512, 512, 512], label='Wide deep clone (3x512) + gentle DAgger'),
    'M55_bc_deep_gentle_o8': dict(_M51, bc_from='O8_lp_grid2', label='Deep clone of the smoother plan O8 + gentle DAgger'),
    'M56_bc_deep_dagger6': dict(_M51, bc_dagger_iters=6, label='Deep clone + gentle DAgger, 6 iterations'),
})
# Round 13: the deep clone (Phase I ~0.687) is close to its plan O2 (0.658), which gives up cost/carbon for smoothness.
# To reach the published top 3 (<= 0.651), clone plans with a better Phase I score, now that deep + gentle DAgger
# copes with less smooth targets (the 64x64 plain clone of O3 managed only 0.797)
METHODS.update({
    'M57_bc_deep_o6': dict(_M51, bc_from='O6_lp_grid05', label='Deep clone of LP plan with grid weight 0.5 (O6) + gentle DAgger'),
    'M58_bc_deep_o3': dict(_M51, bc_from='O3_lp_phase1_eff087', label='Deep clone of Phase I LP plan (O3) + gentle DAgger'),
    'M59_bc_deep_o9': dict(_M51, bc_from='O9_lp_grid025', label='Deep clone of LP plan with grid weight 0.25 (O9) + gentle DAgger'),
    'M60_bc_deep_o6_dagger6': dict(_M51, bc_from='O6_lp_grid05', bc_dagger_iters=6,
                                   label='Deep clone of O6 + gentle DAgger, 6 iterations'),
})
# Round 14: capacity keeps helping with gentle DAgger (seed 42 Phase I / Score: 3x256 0.685 / 0.687, 4x256 0.681 / 0.672,
# 3x512 0.671 / 0.652), and 6 DAgger iterations beat 3 (M49). Combine: bigger networks, more DAgger, better plans
_M54 = METHODS['M54_bc_wide_gentle']
METHODS.update({
    'M61_bc_4x512': dict(_M54, net_arch=[512, 512, 512, 512], label='Clone 4x512 + gentle DAgger'),
    'M62_bc_3x1024': dict(_M54, net_arch=[1024, 1024, 1024], label='Clone 3x1024 + gentle DAgger'),
    'M63_bc_3x512_dagger6': dict(_M54, bc_dagger_iters=6, label='Clone 3x512 + gentle DAgger, 6 iterations'),
    'M64_bc_3x512_o9': dict(_M54, bc_from='O9_lp_grid025', label='Clone 3x512 of LP plan O9 (grid weight 0.25) + gentle DAgger'),
})
# Round 15: the deep clone of the Phase I plan O3 (M58) reached Phase I 0.658 and carbon 0.717 (published #3: 0.651;
# best published carbon on Phase I among the top 2: 0.718 / 0.691). Bigger networks on O3, aimed at the top 3
METHODS.update({
    'M65_bc_3x512_o3': dict(_M54, bc_from='O3_lp_phase1_eff087', label='Clone 3x512 of Phase I LP plan O3 + gentle DAgger'),
    'M66_bc_4x512_o3': dict(_M54, bc_from='O3_lp_phase1_eff087', net_arch=[512, 512, 512, 512],
                            label='Clone 4x512 of Phase I LP plan O3 + gentle DAgger'),
    'M67_bc_3x512_o3_dagger6': dict(_M54, bc_from='O3_lp_phase1_eff087', bc_dagger_iters=6,
                                    label='Clone 3x512 of Phase I LP plan O3 + gentle DAgger, 6 iterations'),
})
RBC_FOLDER = 'B0_RBC'

print(f"Core config loaded. Results folder: {RESULTS_DIR}  |  TOTAL_ENV_STEPS = {TOTAL_ENV_STEPS:,}"
      + ("  [SMOKE TEST]" if SMOKE_TEST else ""))

# %% [markdown]
# ## 2. Per-step terms
#
# After `env.step()`, CityLearn's `time_step` is the index of the hour that was just simulated: its cost uses
# `price[t]` and its emissions use `carbon_intensity[t]`. (Phase 2's dynamic reward read `t - 1`, one hour late;
# fixed here.) Every term is an array of length `n + 1`: one entry per building, and the district in the last slot.
#
# - `ramp`: |load_t - load_{t-1}|
# - `peak`: max(0, load_t - highest load so far today). Over a day these add up to (daily peak - first-hour load),
#   so penalising them pushes the daily peak down. Flaw found in Round 3: a high load in the day's first hour is free
#   and hides later peaks (and the counterfactual reward even pays for it).
# - `peak_full` (methods with `full_day_peak`, from Round 4): as `peak`, but the first hour counts its own load, so a
#   day's terms add up to exactly its peak load.
# - `batt`: |SoC_t - SoC_{t-1}|, a battery-wear proxy (district slot = sum over buildings)

# %%
TERM_NAMES = ['cost', 'carbon', 'ramp', 'peak', 'batt', 'imports', 'load']

def new_tracker(n):
    """Running state needed for ramp / daily-peak / battery terms (reset at the start of each episode)."""
    return {'prev_load': None, 'day_max': None, 'prev_soc': np.zeros(n)}

def step_terms(env, tracker):
    """Cost, carbon, ramp, peak, battery and load terms of the hour just simulated (per building + district)."""
    t = env.time_step
    net = np.array([float(b.net_electricity_consumption[-1]) for b in env.buildings])
    cost = np.array([float(b.net_electricity_consumption_cost[-1]) for b in env.buildings])
    carbon = np.array([float(b.net_electricity_consumption_emission[-1]) for b in env.buildings])
    soc = np.array([float(b.electrical_storage.soc[t]) for b in env.buildings])

    load = np.append(net, net.sum())
    ramp = np.zeros_like(load) if tracker['prev_load'] is None else np.abs(load - tracker['prev_load'])
    if tracker['day_max'] is None or t % 24 == 0:
        peak = np.zeros_like(load)
        peak_full = np.maximum(0.0, load)   # the day's first hour opens its peak, so the day's terms sum to its peak load
        tracker['day_max'] = load.copy()
    else:
        peak = np.maximum(0.0, load - tracker['day_max'])
        peak_full = peak
        tracker['day_max'] = np.maximum(tracker['day_max'], load)
    batt = np.abs(soc - tracker['prev_soc'])
    price = float(env.buildings[0].pricing._electricity_pricing[t])
    ci = float(env.buildings[0].carbon_intensity._carbon_intensity[t])

    tracker['prev_load'] = load
    tracker['prev_soc'] = soc
    imports = np.maximum(net, 0.0)
    return {
        # Official CityLearn 2022 cost / carbon: clipped at 0 on the (district) net load, so exports earn nothing and
        # one building's export offsets another's import (the district slot is what the challenge scores)
        'cost_d': np.maximum(0.0, load * price),
        'carbon_d': np.maximum(0.0, load * ci),
        'cost': np.append(cost, cost.sum()),
        'carbon': np.append(carbon, carbon.sum()),
        'ramp': ramp,
        'peak': peak,
        'peak_full': peak_full,
        'batt': np.append(batt, batt.sum()),
        'imports': np.append(imports, imports.sum()),
        'load': load,
        'load_sq': load ** 2,
        'soc': np.append(soc, soc.mean()),
    }

def trace_row(env, t, terms, actions):
    """One row of an evaluation trace: district totals, building loads / SoC and the actions taken."""
    n = len(env.buildings)
    row = {'step': t}
    for k in ['cost', 'carbon', 'ramp', 'peak', 'batt', 'imports', 'load']:
        row[k] = float(terms[k][n])
    row['carbon_intensity'] = float(env.buildings[0].carbon_intensity._carbon_intensity[t])
    row['pricing'] = float(env.buildings[0].pricing._electricity_pricing[t])
    for i in range(n):
        row[f'load_b{i}'] = float(terms['load'][i])
        row[f'soc_b{i}'] = float(terms['soc'][i])
        row[f'action_b{i}'] = float(actions[i])
    return row

# %% [markdown]
# ## 3. RBC reference
#
# CityLearn's `BasicRBC` is deterministic, so it runs once and is cached in `results/B0_RBC/`. Its per-step terms give
# (a) the normalisers of the multi-KPI reward and (b) the hour-by-hour reference for the adaptive weight updates.

# %%
def compute_self_consumption(buildings):
    """Fraction of PV generation consumed on-site rather than exported to the grid (index 0 is a placeholder)."""
    total_solar = 0.0
    total_exported_solar = 0.0
    for b in buildings:
        solar = np.abs(np.array(b.solar_generation[1:], dtype=np.float64))
        net_cons = np.array(b.net_electricity_consumption[1:], dtype=np.float64)
        exported = np.maximum(0.0, -net_cons)
        total_solar += solar.sum()
        total_exported_solar += np.minimum(exported, solar).sum()
    return (total_solar - total_exported_solar) / total_solar if total_solar > 0 else float('nan')

def run_rbc():
    """Runs the RBC for one full year and caches its trace, per-step terms and KPIs."""
    folder = RESULTS_DIR / RBC_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    env = CityLearnEnv(schema=DATASET_NAME, central_agent=False)
    obs = env.reset()
    agent = BasicRBC(env)
    n = len(env.buildings)
    tracker = new_tracker(n)
    term_arrays = {k: np.zeros((env.time_steps, n + 1)) for k in TERM_NAMES}
    rows = []
    while True:
        actions = agent.predict(obs)
        obs, _, done, _ = env.step(actions)
        t = env.time_step
        terms = step_terms(env, tracker)
        for k in TERM_NAMES:
            term_arrays[k][t] = terms[k]
        rows.append(trace_row(env, t, terms, [a[0] for a in actions]))
        if done:
            break
    trace = pd.DataFrame(rows)
    trace.to_csv(folder / 'eval_trace.csv', index=False)
    np.savez(folder / 'rbc_terms.npz', **term_arrays)
    kpis = compute_kpis(trace, env.buildings)
    with open(folder / 'summary.json', 'w') as f:
        json.dump({'method': RBC_FOLDER, 'seed': None, 'kpis': kpis}, f, indent=2)
    return trace

def full_day_peak(load, steps):
    """The `peak_full` term of step_terms, rebuilt from a cached load series (steps in simulation order)."""
    out, day_max = np.zeros_like(load), None
    for t in steps:
        if day_max is None or t % 24 == 0:
            out[t], day_max = np.maximum(0.0, load[t]), np.array(load[t], dtype=float)
        else:
            out[t], day_max = np.maximum(0.0, load[t] - day_max), np.maximum(day_max, load[t])
    return out

def official_cost_carbon(load, trace):
    """`cost_d` / `carbon_d` terms of step_terms rebuilt from a cached load series and its trace's price / intensity."""
    price, ci = np.zeros(len(load)), np.zeros(len(load))
    price[trace['step'].values], ci[trace['step'].values] = trace['pricing'].values, trace['carbon_intensity'].values
    if load.ndim == 2:
        price, ci = price[:, None], ci[:, None]
    return {'cost_d': np.maximum(0.0, load * price), 'carbon_d': np.maximum(0.0, load * ci)}

def load_rbc():
    """RBC trace, per-step terms (T x (n+1) arrays) and per-term normalisers; runs the RBC first if not cached."""
    folder = RESULTS_DIR / RBC_FOLDER
    if not (folder / 'summary.json').exists():
        print("Running RBC reference (about 1 min, once) ...")
        run_rbc()
    trace = pd.read_csv(folder / 'eval_trace.csv')
    data = np.load(folder / 'rbc_terms.npz')
    terms = {k: data[k] for k in TERM_NAMES}
    terms['load_sq'] = terms['load'] ** 2
    steps = trace['step'].values
    terms['peak_full'] = full_day_peak(terms['load'], steps)
    terms.update(official_cost_carbon(terms['load'], trace))
    # Normaliser of each term = its RBC mean per simulated step (floored to avoid division by ~0)
    norm = {k: np.maximum(np.abs(terms[k][steps].mean(axis=0)), 1e-6) for k in terms}
    return {'trace': trace, 'terms': terms, 'norm': norm}

# %% [markdown]
# ## 4. KPIs (CityLearn-style, as ratios to RBC)
#
# All are computed on the **district** net load `L_t`, lower is better, and the ratio to RBC is reported (RBC = 1.0).
# The **score** used to rank methods is the mean ratio over `SCORE_KPIS`, fixed before any phase 3 result was seen.

# %%
SCORE_KPIS = ['cost', 'carbon', 'ramping', 'daily_peak', 'annual_peak', 'one_minus_load_factor']

def compute_kpis(trace, buildings=None):
    """KPIs of one evaluation trace (district level)."""
    load = trace['load'].values
    n_days = len(load) // 24
    daily_peak = load[:n_days * 24].reshape(n_days, 24).max(axis=1).mean() if n_days else load.max()
    # 1 - load factor per ~month (730 h), averaged, as in CityLearn
    blocks = [load[i:i + 730] for i in range(0, len(load), 730) if len(load[i:i + 730]) >= 24]
    one_minus_lf = float(np.mean([1 - b.mean() / b.max() for b in blocks if b.max() > 0]))
    n_buildings = len([c for c in trace.columns if c.startswith('soc_b')])
    kpis = {
        'cost': float(trace['cost'].sum()),
        'carbon': float(trace['carbon'].sum()),
        'ramping': float(np.abs(np.diff(load)).sum()),
        'daily_peak': float(daily_peak),
        'annual_peak': float(load.max()),
        'one_minus_load_factor': one_minus_lf,
        'electricity_consumption': float(trace['imports'].sum()),
        'zero_net_energy': float(load.sum()),
        'carbon_per_kwh_imported': float(trace['carbon'].sum() / max(trace['imports'].sum(), 1e-9)),
        'battery_cycles': float(trace['batt'].sum() / 2 / max(n_buildings, 1)),   # equivalent full cycles per battery
    }
    if buildings is not None:
        kpis['self_consumption'] = float(compute_self_consumption(buildings))
    return kpis

def kpi_ratios(kpis, rbc_trace, rbc_buildings_kpis=None):
    """Each KPI divided by the RBC's value over the same hours, plus the ranking score."""
    rbc_kpis = compute_kpis(rbc_trace[rbc_trace['step'] <= kpis['_last_step']]) if '_last_step' in kpis else compute_kpis(rbc_trace)
    if rbc_buildings_kpis is not None:
        rbc_kpis['self_consumption'] = rbc_buildings_kpis.get('self_consumption', float('nan'))
    ratios = {k: kpis[k] / rbc_kpis[k] for k in kpis if k in rbc_kpis and not k.startswith('_') and rbc_kpis[k] != 0}
    ratios['score'] = float(np.mean([ratios[k] for k in SCORE_KPIS]))
    return ratios

# %% [markdown]
# ## 5. Rewards
#
# | reward | formula (per building `b`, or the district for the central agent) |
# |---|---|
# | `cost_only` | `-cost` |
# | `static` | `-cost - λ·carbon` |
# | `dynamic` | `-cost - λ·φ(I_t)·carbon`, φ = `linear` (normalised intensity) or `top25` (1 in the dirtiest 25% of hours, else 0) |
# | `multikpi` | `-(ĉost + μ_carbon·φ·ĉarbon + μ_ramp·r̂amp + μ_peak·p̂eak + W_BATT·b̂att)`, hats = divided by RBC per-step mean |
# | `load_squared` (M9) | `-(ĉost + ĉarbon + L̂²)`: squared district load penalises peaks and spikiness smoothly |
#
# Options: `district_penalty` (M10) adds `1/n` of the district ramp + peak terms to each building's simple reward;
# `local_grid_terms=False` (M11) keeps only cost and carbon locally and penalises ramp / peak at district level only.
#
# - Fixed multi-KPI (M2): all μ = 1. Adaptive (M3, M5, M6): μ learned online (section 1 settings), always summing to 3.
# - **Per-building agents** get their own terms, plus `1/n` of the district ramp and peak terms so that the buildings
#   don't all discharge in the same hour. M5 uses one shared set of μ (updated from district KPIs);
#   M6 gives every building its own μ (updated from that building's KPIs vs the RBC's same building).
# - For `cost_only` / `static` / `dynamic`, the per-building rewards add up exactly to the central reward.
# - Round 3 options (central `multikpi`): `mu_fixed` overrides individual fixed weights (M19: carbon 2), `w_batt`
#   overrides `W_BATT` (M18: 0.5), and `counterfactual` (M20) subtracts the same weighted terms of the cached
#   no-battery run at that hour, so the reward measures the improvement over doing nothing.

# %%
def carbon_phi(state, t):
    """Carbon weight scaling phi(I_t) for the hour just simulated."""
    phi = state['cfg'].get('phi', 'none')
    ci = state['ci'][t]
    if phi == 'linear':
        return float(np.clip((ci - state['ci_min']) / (state['ci_max'] - state['ci_min']), 0.0, 1.0))
    if phi == 'top25':
        return 1.0 if ci >= state['ci_q75'] else 0.0
    return 1.0

def compute_rewards(state, terms, t):
    """Reward(s) of the hour just simulated: a float (central) or an array with one entry per building."""
    cfg, n = state['cfg'], state['n']
    phi = carbon_phi(state, t)

    if cfg['reward'] in ('cost_only', 'static', 'dynamic'):
        lam = 0.0 if cfg['reward'] == 'cost_only' else cfg['lam'] * (phi if cfg['reward'] == 'dynamic' else 1.0)
        r = -(terms['cost'] + lam * terms['carbon'])
        if cfg.get('district_penalty'):
            r = r - (terms['ramp'][n] / state['norm']['ramp'][n] + terms['peak'][n] / state['norm']['peak'][n]) / n
    elif cfg['reward'] == 'load_squared':
        h = {k: terms[k] / state['norm'][k] for k in ['cost', 'carbon', 'load_sq']}
        r = -(h['cost'] + h['carbon'] + h['load_sq'])
    else:
        h = {k: terms[k] / state['norm'][k] for k in ['cost', 'carbon', 'ramp', 'peak', 'batt']}
        pk = 'peak_full' if cfg.get('full_day_peak') else 'peak'
        ck, gk = ('cost_d', 'carbon_d') if cfg.get('official_terms') else ('cost', 'carbon')
        h['peak'] = terms[pk] / state['norm'][pk]
        h['cost'], h['carbon'] = terms[ck] / state['norm'][ck], terms[gk] / state['norm'][gk]
        if not cfg.get('adaptive'):
            mu = {k: np.full(n + 1, cfg.get('mu_fixed', {}).get(k, 1.0)) for k in ADAPT_KPIS}
        elif cfg.get('mu_per_building', True):
            mu = state['mu']
        else:
            mu = {k: np.full(n + 1, state['mu'][k][n]) for k in ADAPT_KPIS}
        if cfg['agent'] == 'per_building' and not cfg.get('local_grid_terms', True):
            r = -(h['cost'] + mu['carbon'] * phi * h['carbon'])
        else:
            r = -(h['cost'] + mu['carbon'] * phi * h['carbon'] + mu['ramp'] * h['ramp'] + mu['peak'] * h['peak']
                  + cfg.get('w_batt', W_BATT) * h['batt'])
        if cfg.get('counterfactual'):
            # Reward = improvement over doing nothing this hour: add back the same weighted terms of the no-battery run
            # (which uses no battery, so its batt term is 0). They do not depend on the action, so the optimal policy is
            # unchanged; only the critic's targets get less noisy.
            nb = {'ramp': state['nb_terms']['ramp'][t] / state['norm']['ramp'][n],
                  'peak': state['nb_terms'][pk][t] / state['norm'][pk][n],
                  'cost': state['nb_terms'][ck][t] / state['norm'][ck][n],
                  'carbon': state['nb_terms'][gk][t] / state['norm'][gk][n]}
            r = r + (nb['cost'] + mu['carbon'][n] * phi * nb['carbon'] + mu['ramp'][n] * nb['ramp'] + mu['peak'][n] * nb['peak'])
        if cfg['agent'] == 'per_building':
            district = (state['mu']['ramp'][n] if cfg.get('adaptive') else 1.0) * h['ramp'][n] \
                     + (state['mu']['peak'][n] if cfg.get('adaptive') else 1.0) * h['peak'][n]
            r = r - district / n

    return float(r[n]) if cfg['agent'] == 'central' else r[:n].astype(np.float32)

def update_adaptive_weights(state, terms, t):
    """Accumulates this hour's terms (agent and RBC); once per window, shifts weight toward the KPIs furthest behind."""
    for k in ADAPT_KPIS:
        state['window'][k] += terms[k]
        state['window_rbc'][k] += state['rbc_terms'][k][t]
    state['window_len'] += 1
    if state['window_len'] < ADAPT_WINDOW:
        return
    gaps = []
    for k in ADAPT_KPIS:
        rbc = state['window_rbc'][k]
        ratio = np.where(rbc > 1e-6, state['window'][k] / np.maximum(rbc, 1e-6), ADAPT_TARGETS[k])
        gaps.append(np.log(np.clip(ratio / ADAPT_TARGETS[k], 0.05, 20.0)))
    soft = np.exp(np.array(gaps) / ADAPT_TEMPERATURE)          # shape (n_kpis, n + 1): per building + district
    share = ADAPT_FLOOR / len(ADAPT_KPIS) + (1 - ADAPT_FLOOR) * soft / soft.sum(axis=0)
    new_weights = ADAPT_TOTAL * share
    for i, k in enumerate(ADAPT_KPIS):
        state['mu'][k] = (1 - ADAPT_SMOOTHING) * state['mu'][k] + ADAPT_SMOOTHING * new_weights[i]
        state['window'][k] = np.zeros(state['n'] + 1)
        state['window_rbc'][k] = np.zeros(state['n'] + 1)
    state['window_len'] = 0

# %% [markdown]
# ## 6. Environments
#
# Both environments share one plain-dict `state` (CityLearn env, method config, trackers, adaptive weights) and the
# same `env_step` function, so central and per-building agents compute terms, rewards and weight updates identically.
#
# - **Central** (`CentralEnv`): CityLearn `central_agent=True`, observation = 60 values, action = 5 battery rates.
# - **Per-building** (`PerBuildingVecEnv`): CityLearn `central_agent=False`; SB3 sees the 5 buildings as 5 parallel
#   environments, so ONE PPO policy is shared by all buildings (parameter sharing). Each building observes its own
#   28 values plus a one-hot building ID. All 5 step together in the same simulated district.
# - **Residual** (`residual`, M16, central only): the battery action is the RBC's action for that hour plus the policy's
#   output, clipped to the action bounds. A policy that outputs 0 behaves exactly like the RBC.

# %%
def make_state(env, method, rbc):
    """All mutable per-run data: config, trackers, adaptive weights and episode totals."""
    n = len(env.buildings)
    ci = np.array(env.buildings[0].carbon_intensity._carbon_intensity, dtype=np.float64)
    state = {
        'env': env, 'method': method, 'cfg': METHODS[method], 'n': n,
        'rbc_terms': rbc['terms'], 'norm': rbc['norm'],
        'ci': ci, 'ci_min': float(ci.min()), 'ci_max': float(ci.max()), 'ci_q75': float(np.quantile(ci, 0.75)),
        'mu': {k: np.full(n + 1, MU_INIT) for k in ADAPT_KPIS},
    }
    cfg = METHODS[method]
    if cfg.get('counterfactual') or cfg.get('norm_ref') == 'no_battery':
        state['nb_terms'] = load_no_battery_terms(env.time_steps)
    if cfg.get('norm_ref') == 'no_battery':
        # Score-aligned scaling: each term / its no-battery per-step mean, so the yearly sum of a term is T x its ratio
        # to no battery, the same normalisation as the challenge KPIs (battery wear keeps the RBC scale: it is 0 there)
        steps = rbc['trace']['step'].values
        state['norm'] = dict(state['norm'])
        for k, v in state['nb_terms'].items():
            if k != 'load':
                state['norm'][k] = np.full(n + 1, max(abs(v[steps].mean()), 1e-6))
    return state

def load_cached_actions(name, n, prefix='action_b'):
    """Actions (or, with prefix='soc_b', states of charge) of a cached controller's eval trace by step (row t = the
    action that produced step t / the SoC after it). `name` is a results folder: B0_RBC (trace at its top level) or a run
    like an LP plan in <name>/seed_0/."""
    candidates = [RESULTS_DIR / name / 'eval_trace.csv', RESULTS_DIR / name / 'seed_0' / 'eval_trace.csv',
                  PHASE_DIR / 'results' / name / 'eval_trace.csv', PHASE_DIR / 'results' / name / 'seed_0' / 'eval_trace.csv']
    trace = pd.read_csv(next(p for p in candidates if p.exists()))
    actions = np.zeros((int(trace['step'].max()) + 1, n))
    actions[trace['step'].values] = trace[[f'{prefix}{i}' for i in range(n)]].values
    return actions

def load_no_battery_terms(time_steps):
    """District cost / carbon / ramp / peak of the no-battery run (B5, made by 99_compare_results), indexed by step."""
    # The smoke-test folder has no B5 run, so fall back to the real results folder
    path = next(p for p in [RESULTS_DIR / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv',
                            PHASE_DIR / 'results' / 'B5_no_battery' / 'seed_0' / 'eval_trace.csv'] if p.exists())
    trace = pd.read_csv(path)
    terms = {}
    for k in ['cost', 'carbon', 'ramp', 'peak', 'load']:
        terms[k] = np.zeros(time_steps)
        terms[k][trace['step'].values] = trace[k].values
    terms['peak_full'] = full_day_peak(terms['load'], trace['step'].values)
    terms.update(official_cost_carbon(terms['load'], trace))
    return terms

def reset_state(state, seed):
    """Resets CityLearn and the per-episode trackers (the learned mu carry over between episodes)."""
    if seed is not None:
        state['env'].random_seed = seed
    obs = state['env'].reset()
    n = state['n']
    state['tracker'] = new_tracker(n)
    state['window'] = {k: np.zeros(n + 1) for k in ADAPT_KPIS}
    state['window_rbc'] = {k: np.zeros(n + 1) for k in ADAPT_KPIS}
    state['window_len'] = 0
    state['episode'] = {'cost': 0.0, 'carbon': 0.0, 'reward': 0.0}
    return obs

def env_step(state, citylearn_actions):
    """One CityLearn step: terms, rewards, adaptive weight update and episode totals."""
    obs, _, done, _ = state['env'].step(citylearn_actions)
    t = state['env'].time_step
    terms = step_terms(state['env'], state['tracker'])
    rewards = compute_rewards(state, terms, t)
    if state['cfg'].get('adaptive'):
        update_adaptive_weights(state, terms, t)
    n = state['n']
    state['episode']['cost'] += terms['cost'][n]
    state['episode']['carbon'] += terms['carbon'][n]
    state['episode']['reward'] += float(np.sum(rewards))
    return obs, rewards, done, terms, t

def episode_metrics(state):
    """Episode totals plus the current district-level mu (logged once per training episode)."""
    m = dict(state['episode'])
    for k in ADAPT_KPIS:
        m[f'mu_{k}'] = float(state['mu'][k][state['n']])
    return m


class CentralEnv(gym.Env):
    """One agent controls all batteries (CityLearn central_agent=True), with the method's reward."""
    def __init__(self, method, seed, rbc):
        super().__init__()
        self.env = CityLearnEnv(schema=DATASET_NAME, central_agent=True)
        self.state = make_state(self.env, method, rbc)
        self.seed_value = seed
        old_obs, old_act = self.env.observation_space[0], self.env.action_space[0]
        self.observation_space = gym.spaces.Box(low=old_obs.low, high=old_obs.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=old_act.low, high=old_act.high, dtype=np.float32)
        # Residual methods: base action of each simulated step (row t = action that produced step t), from the RBC cache
        # or, with `residual_base`, from another cached controller's eval trace (e.g. a perfect-foresight LP schedule)
        self.rbc_actions = None
        self.residual_scale = METHODS[method].get('residual_scale', 1.0)
        if METHODS[method].get('residual'):
            self.rbc_actions = load_cached_actions(METHODS[method].get('residual_base', RBC_FOLDER), len(self.env.buildings))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = reset_state(self.state, seed if seed is not None else self.seed_value)
        return np.array(obs[0], dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, terms, t = env_step(self.state, self.to_citylearn_actions(action))
        info = {}
        if done:
            info['episode_metrics'] = episode_metrics(self.state)
        return np.array(obs[0], dtype=np.float32), reward, done, False, info

    def to_citylearn_actions(self, action):
        if self.rbc_actions is None:
            return [action]
        t_next = min(self.env.time_step + 1, len(self.rbc_actions) - 1)   # the step this action will produce
        return [np.clip(self.residual_scale * action + self.rbc_actions[t_next], self.action_space.low, self.action_space.high)]

    def policy_obs(self, citylearn_obs):
        return np.array(citylearn_obs[0], dtype=np.float32)


class PerBuildingVecEnv(VecEnv):
    """The 5 buildings of ONE CityLearn district exposed to SB3 as 5 parallel envs (one shared policy)."""
    def __init__(self, method, seed, rbc):
        self.env = CityLearnEnv(schema=DATASET_NAME, central_agent=False)
        self.state = make_state(self.env, method, rbc)
        self.seed_value = seed
        n = self.state['n']
        lows = np.min([s.low for s in self.env.observation_space], axis=0)
        highs = np.max([s.high for s in self.env.observation_space], axis=0)
        obs_space = gym.spaces.Box(low=np.append(lows, np.zeros(n)).astype(np.float32),
                                   high=np.append(highs, np.ones(n)).astype(np.float32), dtype=np.float32)
        act = self.env.action_space[0]
        act_space = gym.spaces.Box(low=act.low, high=act.high, dtype=np.float32)
        super().__init__(n, obs_space, act_space)

    def policy_obs(self, citylearn_obs):
        n = self.state['n']
        return np.array([np.concatenate([o, np.eye(n)[i]]) for i, o in enumerate(citylearn_obs)], dtype=np.float32)

    def to_citylearn_actions(self, actions):
        return [[float(a[0])] for a in actions]

    def reset(self):
        return self.policy_obs(reset_state(self.state, self.seed_value))

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, done, terms, t = env_step(self.state, self.to_citylearn_actions(self._actions))
        obs = self.policy_obs(obs)
        infos = [{} for _ in range(self.num_envs)]
        dones = np.full(self.num_envs, done)
        if done:
            infos[0]['episode_metrics'] = episode_metrics(self.state)
            for i in range(self.num_envs):
                infos[i]['terminal_observation'] = obs[i]
            obs = self.reset()
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

def make_env(method, seed, rbc):
    """Central or per-building environment, depending on the method."""
    if METHODS[method]['agent'] == 'per_building':
        return PerBuildingVecEnv(method, seed, rbc)
    return CentralEnv(method, seed, rbc)

# %% [markdown]
# ## 7. Resumable training, evaluation and `run_method`
#
# - Every `CHECKPOINT_EVERY` env steps the model, the learned μ and the training log are saved to
#   `checkpoint.zip` / `checkpoint.json`. Re-running a notebook continues from the last checkpoint
#   (the episode restarts from January; PPO's step counter, weights and μ continue).
# - When training is complete, `model.zip` is written; evaluation then writes `eval_trace.csv` and `summary.json`.
# - `run_method(METHOD, SEEDS)` skips finished seeds, resumes unfinished ones and starts new ones.
# - `normalize` methods (M7, M8) wrap the env in SB3's `VecNormalize` (running mean/std of observations and rewards);
#   its statistics are saved with every checkpoint (`vecnormalize.pkl`) and reused, frozen, for evaluation.
# - SAC (M12) also saves its replay buffer with every checkpoint (`replay_buffer.pkl`), so a resumed run keeps its data.

# %%
def run_dir_of(method, seed):
    return RESULTS_DIR / method / f'seed_{seed}'

def save_checkpoint(model, run_dir, state, log_rows, env_steps, env=None):
    """Model + mu + training log (+ VecNormalize stats, SAC replay buffer), written atomically (temp file, then rename)."""
    model.save(run_dir / 'checkpoint_tmp.zip')
    os.replace(run_dir / 'checkpoint_tmp.zip', run_dir / 'checkpoint.zip')
    if isinstance(env, VecNormalize):
        env.save(run_dir / 'vecnormalize_tmp.pkl')
        os.replace(run_dir / 'vecnormalize_tmp.pkl', run_dir / 'vecnormalize.pkl')
    if isinstance(model, SAC):
        model.save_replay_buffer(run_dir / 'replay_buffer_tmp.pkl')
        os.replace(run_dir / 'replay_buffer_tmp.pkl', run_dir / 'replay_buffer.pkl')
    info = {'env_steps': env_steps, 'mu': {k: state['mu'][k].tolist() for k in ADAPT_KPIS}, 'log': log_rows}
    with open(run_dir / 'checkpoint_tmp.json', 'w') as f:
        json.dump(info, f)
    os.replace(run_dir / 'checkpoint_tmp.json', run_dir / 'checkpoint.json')


class TrainCallback(BaseCallback):
    """Logs one row per training episode and saves a resumable checkpoint every CHECKPOINT_EVERY env steps."""
    def __init__(self, run_dir, state, n_envs, log_rows, start_time, env):
        super().__init__(verbose=0)
        self.run_dir, self.state, self.n_envs, self.log_rows, self.env = run_dir, state, n_envs, log_rows, env
        self.start_time = start_time
        self.last_save = None

    def env_steps(self):
        return self.num_timesteps // self.n_envs

    def _on_step(self):
        if self.last_save is None:
            self.last_save = self.env_steps()
        for info in self.locals.get('infos', []):
            if 'episode_metrics' in info:
                row = {'episode': len(self.log_rows) + 1, 'env_step': self.env_steps(), **info['episode_metrics']}
                self.log_rows.append(row)
                print(f"  episode {row['episode']:>3} | env step {row['env_step']:>7,} | cost {row['cost']:8.1f}"
                      f" | carbon {row['carbon']:8.1f} | mu(c,r,p) = ({row['mu_carbon']:.2f}, {row['mu_ramp']:.2f},"
                      f" {row['mu_peak']:.2f}) | {(time.time() - self.start_time) / 60:.1f} min")
        return True

    def _on_rollout_end(self):
        if self.env_steps() - self.last_save >= CHECKPOINT_EVERY:
            save_checkpoint(self.model, self.run_dir, self.state, self.log_rows, self.env_steps(), self.env)
            self.last_save = self.env_steps()
            print(f"  [checkpoint saved at env step {self.env_steps():,} / {total_env_steps(self.state['method']):,}]")


def total_env_steps(method):
    """Training length of a method: TOTAL_ENV_STEPS unless the method sets `env_steps` (the smoke test always uses its own)."""
    return TOTAL_ENV_STEPS if SMOKE_TEST else METHODS[method].get('env_steps', TOTAL_ENV_STEPS)


def sac_settings(method):
    """SAC hyperparameters and policy kwargs, with the method's optional `sac_params` / `net_arch` overrides."""
    cfg = METHODS[method]
    return dict(SAC_PARAMS, **cfg.get('sac_params', {})), {'net_arch': cfg.get('net_arch', SAC_POLICY_KWARGS['net_arch'])}


def make_training_env(method, base, run_dir):
    """The env SB3 trains on: the base env, wrapped in VecNormalize (restored from the run's stats) if requested."""
    if not METHODS[method].get('normalize'):
        return base
    venv = base if isinstance(base, VecEnv) else DummyVecEnv([lambda: base])
    if (run_dir / 'vecnormalize.pkl').exists():
        return VecNormalize.load(run_dir / 'vecnormalize.pkl', venv)
    gamma = sac_settings(method)[0]['gamma'] if METHODS[method].get('algo') == 'sac' else PPO_PARAMS['gamma']
    return VecNormalize(venv, norm_obs=True, norm_reward=METHODS[method].get('normalize_reward', True), clip_obs=10.0,
                        gamma=gamma)


BC_EPOCHS = 30 if SMOKE_TEST else 300

def bc_pretrain(model, method, base, env):
    """Behaviour cloning: fits the SAC actor's deterministic action to a cached controller's actions (`bc_from`, e.g. a
    perfect-foresight LP plan), on the observations seen along that controller's own trajectory. The cloned policy only
    needs the current observation, not the plan. With VecNormalize, the observation statistics are set from that data."""
    plan = load_cached_actions(METHODS[method]['bc_from'], len(base.env.buildings))
    obs_list, act_list = [], []
    obs = base.policy_obs(reset_state(base.state, base.seed_value))
    while True:
        a = plan[min(base.env.time_step + 1, len(plan) - 1)].astype(np.float32)
        obs_list.append(obs)
        act_list.append(a)
        cl_obs, _, done, _, _ = env_step(base.state, [a])
        obs = base.policy_obs(cl_obs)
        if done:
            break
    X_raw = np.array(obs_list, dtype=np.float32)
    if isinstance(env, VecNormalize):
        env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count = X_raw.mean(0), X_raw.var(0) + 1e-8, float(len(X_raw))
    scale = env.normalize_obs if isinstance(env, VecNormalize) else (lambda o: o)
    to_unit = lambda a: np.clip(model.policy.scale_action(np.asarray(a)), -0.999, 0.999)   # actor's [-1, 1] range
    X, Y = scale(X_raw), to_unit(act_list)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)

    def fit(X, Y, epochs):
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        for epoch in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                idx = perm[i:i + 256]
                loss = torch.nn.functional.mse_loss(model.actor(Xt[idx], deterministic=True), Yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return torch.nn.functional.mse_loss(model.actor(Xt, deterministic=True), Yt).item()

    final = fit(X, Y, BC_EPOCHS)
    print(f"  behaviour cloning from {METHODS[method]['bc_from']}: {len(X):,} samples, {BC_EPOCHS} epochs, "
          f"final MSE {final:.4f} (actions scaled to [-1, 1])")
    # DAgger: roll the clone out, label every state it visits with the expert's action there, aggregate, refit. The expert
    # is the plan's action for that hour plus a correction of `bc_dagger_gain` x the state-of-charge error (gain 1 = catch
    # up in one hour, which Round 8 showed makes the clone jumpy; smaller gains recover over several hours)
    plan_soc = load_cached_actions(METHODS[method]['bc_from'], len(base.env.buildings), prefix='soc_b')
    gain = METHODS[method].get('bc_dagger_gain', 1.0)
    eta = 0.9 ** 0.5
    for it in range(METHODS[method].get('bc_dagger_iters', 0)):
        obs, new_X, new_Y, done = base.policy_obs(reset_state(base.state, base.seed_value)), [], [], False
        while not done:
            t_next = min(base.env.time_step + 1, len(plan_soc) - 1)
            cur = np.array([b.electrical_storage.soc[b.electrical_storage.time_step] for b in base.env.buildings])
            error = plan_soc[t_next - 1] - cur                                   # how far below the plan's SoC we are now
            correction = gain * np.where(error >= 0, error / eta, error * eta)
            expert = np.clip(plan[t_next] + correction, -0.78125, 0.78125)
            new_X.append(obs)
            new_Y.append(expert)
            with torch.no_grad():
                a = model.actor(torch.tensor(scale(obs[None]), dtype=torch.float32), deterministic=True).numpy()[0]
            cl_obs, _, done, _, _ = env_step(base.state, [model.policy.unscale_action(a).astype(np.float32)])
            obs = base.policy_obs(cl_obs)
        X, Y = np.concatenate([X, scale(np.array(new_X, dtype=np.float32))]), np.concatenate([Y, to_unit(new_Y)])
        final = fit(X, Y, max(BC_EPOCHS // 3, 1))
        print(f"  DAgger iteration {it + 1}: {len(X):,} samples, MSE {final:.4f}")


def train_run(method, seed, rbc):
    """Trains (or resumes, or loads) one method x seed. Returns (model, base env, training env)."""
    run_dir = run_dir_of(method, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = METHODS[method]
    algo = SAC if cfg.get('algo') == 'sac' else PPO
    base = make_env(method, seed, rbc)
    env = make_training_env(method, base, run_dir)
    n_envs = base.num_envs if isinstance(base, VecEnv) else 1

    if (run_dir / 'model.zip').exists():
        print("  training already finished - loading model.zip")
        model = algo.load(run_dir / 'model.zip', env=env, device='cpu')
        with open(run_dir / 'checkpoint.json') as f:
            base.state['mu'] = {k: np.array(v) for k, v in json.load(f)['mu'].items()}
        return model, base, env

    if algo is SAC:
        sac_params, sac_policy_kwargs = sac_settings(method)
        new_model = lambda: SAC('MlpPolicy', env, policy_kwargs=sac_policy_kwargs, seed=seed, device='cpu', **sac_params)
    else:
        # Per-building: 5 samples per env step, so batch size x5 keeps the number of gradient steps equal to central
        params = dict(PPO_PARAMS, batch_size=PPO_PARAMS['batch_size'] * n_envs)
        new_model = lambda: PPO('MlpPolicy', env, policy_kwargs=POLICY_KWARGS, seed=seed, device='cpu', **params)

    if (run_dir / 'checkpoint.zip').exists():
        with open(run_dir / 'checkpoint.json') as f:
            ckpt = json.load(f)
        model = algo.load(run_dir / 'checkpoint.zip', env=env, device='cpu')
        if algo is SAC and (run_dir / 'replay_buffer.pkl').exists():
            model.load_replay_buffer(run_dir / 'replay_buffer.pkl')
        base.state['mu'] = {k: np.array(v) for k, v in ckpt['mu'].items()}
        log_rows = ckpt['log']
        print(f"  resuming from checkpoint at env step {ckpt['env_steps']:,}")
    else:
        model = new_model()
        log_rows = []
        if cfg.get('bc_from'):
            bc_pretrain(model, method, base, env)

    remaining = total_env_steps(method) * n_envs - model.num_timesteps
    if remaining > 0:
        callback = TrainCallback(run_dir, base.state, n_envs, log_rows, time.time(), env)
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    save_checkpoint(model, run_dir, base.state, log_rows, model.num_timesteps // n_envs, env)
    model.save(run_dir / 'model.zip')
    pd.DataFrame(log_rows).to_csv(run_dir / 'train_log.csv', index=False)
    return model, base, env


def evaluate_run(method, seed, model, base, env, rbc):
    """Deterministic policy over the full year (EVAL_STEPS in smoke test); writes eval_trace.csv and summary.json."""
    state = base.state
    mu_trained = {k: state['mu'][k].tolist() for k in ADAPT_KPIS}
    # VecNormalize methods: the policy sees observations scaled with the (frozen) training statistics
    scale = env.normalize_obs if isinstance(env, VecNormalize) else (lambda o: o)
    if isinstance(env, VecNormalize):
        env.training = False
    obs = scale(base.policy_obs(reset_state(state, seed)))
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        cl_actions = base.to_citylearn_actions(action)
        cl_obs, _, done, terms, t = env_step(state, cl_actions)
        flat = np.array(cl_actions[0]) if METHODS[method]['agent'] == 'central' else np.array([a[0] for a in cl_actions])
        rows.append(trace_row(state['env'], t, terms, flat))
        obs = scale(base.policy_obs(cl_obs))
        if done or (EVAL_STEPS is not None and len(rows) >= EVAL_STEPS):
            break

    trace = pd.DataFrame(rows)
    kpis = compute_kpis(trace, state['env'].buildings)
    kpis['_last_step'] = int(trace['step'].max())
    with open(RESULTS_DIR / RBC_FOLDER / 'summary.json') as f:
        rbc_summary = json.load(f)
    ratios = kpi_ratios(kpis, rbc['trace'], rbc_summary['kpis'])
    kpis.pop('_last_step')
    summary = {'method': method, 'label': METHODS[method]['label'], 'seed': seed,
               'env_steps_trained': total_env_steps(method), 'kpis': kpis, 'ratios_vs_rbc': ratios, 'mu_final': mu_trained}
    run_dir = run_dir_of(method, seed)
    trace.to_csv(run_dir / 'eval_trace.csv', index=False)
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def summary_row(summary):
    """Flat one-line view of a summary.json: ratios to RBC (lower is better) and raw cost / carbon."""
    row = {'method': summary['method'], 'seed': summary['seed'], 'score': summary['ratios_vs_rbc']['score']}
    row.update({f'{k}_ratio': v for k, v in summary['ratios_vs_rbc'].items() if k != 'score'})
    row['cost'] = summary['kpis']['cost']
    row['carbon'] = summary['kpis']['carbon']
    return row


def run_method(method, seeds):
    """Trains + evaluates every seed of one method; finished seeds are skipped, unfinished ones resume."""
    rbc = load_rbc()
    rows = []
    for seed in seeds:
        run_dir = run_dir_of(method, seed)
        if (run_dir / 'summary.json').exists():
            print(f"[{method} | seed {seed}] already finished - skipping")
            with open(run_dir / 'summary.json') as f:
                rows.append(summary_row(json.load(f)))
            continue
        print(f"\n[{method} | seed {seed}] training for {total_env_steps(method):,} env steps ...")
        start = time.time()
        model, base, env = train_run(method, seed, rbc)
        print(f"[{method} | seed {seed}] evaluating ...")
        summary = evaluate_run(method, seed, model, base, env, rbc)
        rows.append(summary_row(summary))
        print(f"[{method} | seed {seed}] done in {(time.time() - start) / 60:.1f} min | score {summary['ratios_vs_rbc']['score']:.4f}"
              f" | cost ratio {summary['ratios_vs_rbc']['cost']:.4f} | carbon ratio {summary['ratios_vs_rbc']['carbon']:.4f}")
    return pd.DataFrame(rows)

print("Core functions defined.")
