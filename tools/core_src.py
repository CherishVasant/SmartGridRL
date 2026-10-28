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
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib import TQC, CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

warnings.filterwarnings('ignore')

# One thread per run: small MLPs train as fast on 1 thread, and several notebooks can run side by side
torch.set_num_threads(1)

# %% [markdown]
# ## 1. Configuration

# %%
# The project folder (holding tools/, results/, code/) is found by its tools/core_src.py, so the notebooks work from any
# start folder: the project root, code/ or code/Round_XX/
_cwd = Path.cwd().resolve()
PHASE_DIR = next((p for p in [_cwd, *_cwd.parents] if (p / 'tools' / 'core_src.py').exists()), _cwd)

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
# Round 16 (queued before Round 14/15 results, combining the three levers that each helped: bigger networks, 6 DAgger
# iterations (M49 > M41, M56 > M51, M60 > M57), and the plan: O3 for Phase I, O9 balanced, O2 for the full score)
METHODS.update({
    'M68_bc_4x512_o3_dagger6': dict(_M54, bc_from='O3_lp_phase1_eff087', net_arch=[512, 512, 512, 512], bc_dagger_iters=6,
                                    label='Clone 4x512 of Phase I LP plan O3 + gentle DAgger, 6 iterations'),
    'M69_bc_4x512_o9_dagger6': dict(_M54, bc_from='O9_lp_grid025', net_arch=[512, 512, 512, 512], bc_dagger_iters=6,
                                    label='Clone 4x512 of LP plan O9 + gentle DAgger, 6 iterations'),
    'M70_bc_4x512_o2_dagger6': dict(_M54, net_arch=[512, 512, 512, 512], bc_dagger_iters=6,
                                    label='Clone 4x512 of grid LP plan O2 + gentle DAgger, 6 iterations'),
    'M71_bc_3x512_o3_dagger10': dict(_M54, bc_from='O3_lp_phase1_eff087', bc_dagger_iters=10,
                                     label='Clone 3x512 of Phase I LP plan O3 + gentle DAgger, 10 iterations'),
})
# Round 17: the 3x512 clone of O9 (M64, seed 42) reached Phase I 0.649 (published #3: 0.651) with Score 0.653. Clone the
# new plans around O9: O10 (grid weight 0.1: Phase I 0.634, Score 0.645) and O11 (grid 0.25, carbon 2: G 0.701)
METHODS.update({
    'M72_bc_3x512_o10_dagger6': dict(_M54, bc_from='O10_lp_grid01', bc_dagger_iters=6,
                                     label='Clone 3x512 of LP plan O10 (grid weight 0.1) + gentle DAgger, 6 iterations'),
    'M73_bc_3x512_o11_dagger6': dict(_M54, bc_from='O11_lp_grid025_carbon2', bc_dagger_iters=6,
                                     label='Clone 3x512 of LP plan O11 (grid 0.25, carbon 2) + gentle DAgger, 6 iterations'),
    'M74_bc_4x512_o10_dagger6': dict(_M54, bc_from='O10_lp_grid01', net_arch=[512, 512, 512, 512], bc_dagger_iters=6,
                                     label='Clone 4x512 of LP plan O10 + gentle DAgger, 6 iterations'),
    'M75_bc_3x512_o9_dagger6': dict(_M54, bc_from='O9_lp_grid025', bc_dagger_iters=6,
                                    label='Clone 3x512 of LP plan O9 + gentle DAgger, 6 iterations'),
})
# Round 18: the 4x512 clone of O3 (M66) is a robust published #3 (seeds 42 / 100: Phase I 0.642 / 0.639, G 0.703 / 0.701),
# 0.005 behind Together (0.635) and ~0.011 above its plan O3 (0.629). Push the precision of the closed-loop imitation
_O3 = dict(_M54, bc_from='O3_lp_phase1_eff087')
METHODS.update({
    'M76_bc_5x512_o3_dagger6': dict(_O3, net_arch=[512] * 5, bc_dagger_iters=6,
                                    label='Clone 5x512 of Phase I LP plan O3 + gentle DAgger, 6 iterations'),
    'M77_bc_4x1024_o3_dagger6': dict(_O3, net_arch=[1024] * 4, bc_dagger_iters=6,
                                     label='Clone 4x1024 of Phase I LP plan O3 + gentle DAgger, 6 iterations'),
    'M78_bc_4x512_o3_dagger10': dict(_O3, net_arch=[512] * 4, bc_dagger_iters=10,
                                     label='Clone 4x512 of Phase I LP plan O3 + gentle DAgger, 10 iterations'),
    'M79_bc_4x512_o3_gain02': dict(_O3, net_arch=[512] * 4, bc_dagger_iters=6, bc_dagger_gain=0.2,
                                   label='Clone 4x512 of Phase I LP plan O3 + DAgger gain 0.2, 6 iterations'),
})
# Round 19 (generalisation): on the held-out phase_2 buildings the smooth clones keep C ~0.74 / G ~0.82 but lose the grid
# benefit (D ~1.0); Phase III public formula ~0.774 vs Together's 0.728. Regularise the clone of O9 (M75) so it relies
# less on exact input values: Gaussian input noise during cloning, and a smaller network
_M75 = METHODS['M75_bc_3x512_o9_dagger6']
METHODS.update({
    'M80_bc_o9_noise01': dict(_M75, bc_obs_noise=0.1, label='Clone 3x512 of O9 + DAgger 6, input noise 0.1'),
    'M81_bc_o9_noise03': dict(_M75, bc_obs_noise=0.3, label='Clone 3x512 of O9 + DAgger 6, input noise 0.3'),
    'M82_bc_o9_small_noise01': dict(_M75, net_arch=[256, 256], bc_obs_noise=0.1,
                                    label='Clone 2x256 of O9 + DAgger 6, input noise 0.1'),
})
# Round 20: DAgger iterations keep paying off on the Phase I plan (3x512 clone of O3, seed 42: 3 -> 0.650, 6 -> 0.643,
# 10 -> 0.637; Together 0.635), while network size has saturated. Push the number of DAgger iterations further
METHODS.update({
    'M83_bc_3x512_o3_dagger15': dict(_O3, bc_dagger_iters=15, label='Clone 3x512 of Phase I LP plan O3 + DAgger, 15 iterations'),
    'M84_bc_3x512_o3_dagger20': dict(_O3, bc_dagger_iters=20, label='Clone 3x512 of Phase I LP plan O3 + DAgger, 20 iterations'),
    'M85_bc_4x512_o3_dagger15': dict(_O3, net_arch=[512] * 4, bc_dagger_iters=15,
                                     label='Clone 4x512 of Phase I LP plan O3 + DAgger, 15 iterations'),
    'M86_bc_3x512_o10_dagger15': dict(_M54, bc_from='O10_lp_grid01', bc_dagger_iters=15,
                                      label='Clone 3x512 of LP plan O10 + DAgger, 15 iterations'),
})
# Round 21 (generalisation): building-order augmentation. A smaller clone + input noise (M82) did not transfer better
# (held-out mean(C, G, D) 0.888 vs M75's 0.861). Instead, shuffle the 5 building slots in the cloning data
METHODS.update({
    'M87_bc_o9_permute2': dict(_M75, bc_permute=2, label='Clone 3x512 of O9 + DAgger 6, building-order augmentation x2'),
    'M88_bc_4x512_o9_permute2': dict(METHODS['M69_bc_4x512_o9_dagger6'], bc_permute=2,
                                     label='Clone 4x512 of O9 + DAgger 6, building-order augmentation x2'),
    'M89_bc_o9_permute4': dict(_M75, bc_permute=4, label='Clone 3x512 of O9 + DAgger 6, building-order augmentation x4'),
})
# Round 22 (from 2026-10-19): genuine RL only, deployable in the real world. No perfect-foresight plans, no cloning: the
# policy learns from reward alone and sees only the present, the past and CityLearn's own forecasts. Explore on seed 42,
# one change at a time from the best RL run so far (M27: SAC, score-aligned reward, carbon 2; seed 42 Phase I 0.767),
# then stack what works and only then run more seeds.
_M27 = METHODS['M27_sac_score_aligned_carbon2']
METHODS.update({
    'M90_sac_c2_256': dict(_M27, net_arch=[256, 256], label='SAC score-aligned carbon 2, 256x256'),
    'M91_tqc_c2_256': dict(_M27, algo='tqc', net_arch=[256, 256], label='TQC score-aligned carbon 2, 256x256'),
    'M92_crossq_c2_256': dict(_M27, algo='crossq', net_arch=[256, 256], label='CrossQ score-aligned carbon 2, 256x256'),
    'M93_sac_c2_history': dict(_M27, history_features=True,
                               label='SAC score-aligned carbon 2 + history features (cyclic time, yesterday persistence)'),
    'M94_sac_c2_nstep3': dict(_M27, sac_params={'n_steps': 3}, label='SAC score-aligned carbon 2, 3-step returns'),
    'M95_sac_c2_gamma095': dict(_M27, sac_params={'gamma': 0.95}, label='SAC score-aligned carbon 2, gamma 0.95'),
    'M96_sac_c2_256_history': dict(_M27, net_arch=[256, 256], history_features=True,
                                   label='SAC score-aligned carbon 2, 256x256 + history features'),
    'M97_tqc_c2_256_history': dict(_M27, algo='tqc', net_arch=[256, 256], history_features=True,
                                   label='TQC score-aligned carbon 2, 256x256 + history features'),
})
# Round 22b (same seed, added while Round 22 runs, once the imitation runs freed CPU): more single changes from M27
METHODS.update({
    'M98_sac_c2_per_building': dict(_M27, agent='per_building', sac_params={'buffer_size': 1_000_000},
                                    label='SAC score-aligned carbon 2, one shared policy per building'),
    'M99_sac_c2_utd2': dict(_M27, sac_params={'gradient_steps': 2}, label='SAC score-aligned carbon 2, 2 updates per step'),
    'M100_crossq_c2_256_history': dict(_M27, algo='crossq', net_arch=[256, 256], history_features=True,
                                       label='CrossQ score-aligned carbon 2, 256x256 + history features'),
    'M101_sac_c3': dict(_M27, mu_fixed={'carbon': 3.0, 'ramp': 0.5, 'peak': 0.5}, label='SAC score-aligned, carbon 3'),
})
# 2026-10-19 19:30 (user: paper deadline): Round 22 shortened to 60k env steps (~7 simulated years; M27's training reward
# had plateaued by then), with M102 = M27 at the same 60k budget as the fair control
ROUND22_STEPS = 60_000
METHODS['M102_sac_c2_60k'] = dict(_M27, label='SAC score-aligned carbon 2 (M27) at 60k steps, control for Round 22')
for _m in ['M90_sac_c2_256', 'M91_tqc_c2_256', 'M92_crossq_c2_256', 'M93_sac_c2_history', 'M94_sac_c2_nstep3',
           'M95_sac_c2_gamma095', 'M96_sac_c2_256_history', 'M97_tqc_c2_256_history', 'M98_sac_c2_per_building',
           'M99_sac_c2_utd2', 'M100_crossq_c2_256_history', 'M101_sac_c3', 'M102_sac_c2_60k']:
    METHODS[_m]['env_steps'] = ROUND22_STEPS
# Round 23 (seed 42, 60k steps): build on M93 (history features; seed 42 Phase I 0.774 at 60k vs 0.767 for M27 at 200k).
# Bigger networks (M90) and 2 updates per step (M99) were worse at this budget (0.809), so: better inputs and tuning
_M93 = METHODS['M93_sac_c2_history']
METHODS.update({
    'M103_sac_hist_rich': dict(_M93, history_features='rich',
                               label='SAC carbon 2 + rich history (per-building lags, 24 h ago, yesterday stats)'),
    'M104_sac_hist_dense': dict(_M93, history_features='dense', label='SAC carbon 2 + dense history (next 1-12 h)'),
    'M105_sac_hist_obsnorm': dict(_M93, normalize=True, normalize_reward=False,
                                  label='SAC carbon 2 + history + observation normalisation'),
    'M106_sac_hist_lr1e3': dict(_M93, sac_params={'learning_rate': 1e-3}, label='SAC carbon 2 + history, lr 1e-3'),
    'M107_sac_hist_batch512': dict(_M93, sac_params={'batch_size': 512}, label='SAC carbon 2 + history, batch 512'),
    'M108_sac_hist_tau002': dict(_M93, sac_params={'tau': 0.02}, label='SAC carbon 2 + history, tau 0.02'),
})
# Longer training: M27 improved from 60k (M102, 0.794 +- 0.016) to 200k steps (0.768 +- 0.005). M109 continues each M27
# seed from its 200k checkpoint and replay buffer (copied into results/M109.../seed_*) to 400k env steps
METHODS['M109_sac_c2_400k'] = dict(_M27, env_steps=400_000,
                                   label='SAC score-aligned carbon 2 (M27) trained to 400k env steps')
# TQC was the best new algorithm at 60k (M91, seed 42: 0.794 vs control 0.803): full-budget test with M27's exact settings
METHODS['M110_tqc_c2'] = dict(_M27, algo='tqc', label='TQC score-aligned carbon 2 (M27 settings, 64x64, 200k steps)')
# Full-budget (200k, 3 seeds) tests of the history features: at 60k single-seed differences were within noise
# (Round 23 seed 42: M105 rich-free history + obs norm 0.787, M103 rich history 0.789, control M102 0.803 +- 0.016)
METHODS['M111_sac_hist_rich_obsnorm'] = dict(_M27, history_features='rich', normalize=True, normalize_reward=False,
                                             label='SAC carbon 2 + rich history + observation normalisation, 200k')
METHODS['M112_sac_hist_200k'] = dict(_M27, history_features=True, label='SAC carbon 2 + history features (M93), 200k')
# The two best full-budget changes stacked: TQC (M110, best score 0.861) + history features (M112, best transfer D)
METHODS['M113_tqc_hist'] = dict(_M27, algo='tqc', history_features=True, label='TQC carbon 2 + history features, 200k')
METHODS['M114_tqc_score_aligned'] = dict(METHODS['M26_sac_score_aligned'], algo='tqc',
                                         label='TQC score-aligned (M26 reward, carbon weight 1), 200k')
# Round 25 (2026-10-20): improve the best controller M113 (TQC + history, Phase I 0.763 +- 0.005). Full budget, 3 seeds
# each from the start (60k single-seed screening proved too noisy). One change each:
_M113 = METHODS['M113_tqc_hist']
METHODS.update({
    'M115_tqc_hist_5critics': dict(_M113, policy_kwargs={'n_critics': 5},
                                   label='TQC + history, 5 critics (TQC paper default)'),
    'M116_tqc_hist_lrdecay300k': dict(_M113, lr_decay=True, env_steps=300_000,
                                      label='TQC + history, learning rate decayed linearly to 0 over 300k steps'),
    'M117_tqc_hist_drop5': dict(_M113, sac_params={'top_quantiles_to_drop_per_net': 5},
                                label='TQC + history, 5 of 25 quantiles dropped per critic (more conservative)'),
    'M118_tqc_hist_utd2': dict(_M113, sac_params={'gradient_steps': 2}, label='TQC + history, 2 updates per env step'),
    'M119_tqc_hist_ent10': dict(_M113, sac_params={'target_entropy': -10.0},
                                label='TQC + history, target entropy -10 (less exploration noise)'),
})
# Round 25b (user: one seed only until told to stop; then 3+ seeds for the top models): more single changes to M113
METHODS.update({
    'M120_tqc_hist_phase1': dict(_M113, mu_fixed={'carbon': 2.0, 'ramp': 0.2, 'peak': 0.2},
                                 label='TQC + history, Phase I-focused reward (grid weights 0.2)'),
    'M121_tqc_hist_c15': dict(_M113, mu_fixed={'carbon': 1.5, 'ramp': 0.5, 'peak': 0.5},
                              label='TQC + history, carbon weight 1.5'),
    'M122_tqc_hist_256': dict(_M113, net_arch=[256, 256], label='TQC + history, 256x256 networks'),
    'M123_tqc_hist_gamma098': dict(_M113, sac_params={'gamma': 0.98}, label='TQC + history, gamma 0.98'),
    'M124_tqc_hist_batch512': dict(_M113, sac_params={'batch_size': 512}, label='TQC + history, batch 512'),
    'M125_tqc_hist_rich': dict(_M113, history_features='rich', label='TQC + rich history features'),
    'M126_tqc_hist_dense': dict(_M113, history_features='dense', label='TQC + dense history features'),
    'M127_tqc_hist_nowear': dict(_M113, w_batt=0.0, label='TQC + history, no battery-wear penalty'),
    'M128_tqc_hist_wear03': dict(_M113, w_batt=0.3, label='TQC + history, battery-wear weight 0.3'),
})
# Round 26 (2026-10-20): forecasting + optimisation + RL, as the top challenge teams did, but fully causal. Plain MPC
# with causal forecasts and a ramp weight already beats every pure RL controller (tools/mpc_eval.py: ramp weight 2 ->
# Phase I 0.745, score 0.811 vs M113 0.763 / 0.862). Here RL sets the MPC's weights every hour (see CentralEnv).
METHODS.update({
    'M129_rl_tuned_mpc': dict(_M27, algo='tqc', env_steps=100_000,
                              mpc_tune={'horizon': 24, 'w_carbon': 1.0, 'w_grid': 2.0},
                              label='RL-tuned MPC: TQC sets the weights of a causal-forecast MPC every hour'),
    'M130_rl_tuned_mpc_sac': dict(_M27, env_steps=100_000, mpc_tune={'horizon': 24, 'w_carbon': 1.0, 'w_grid': 2.0},
                                  label='RL-tuned MPC with SAC'),
})
# The full recipe of the top teams, leakage-free: LightGBM forecasts (trained on other buildings) + MPC + RL tuning it
# M129 traded grid smoothness for cost (Phase III public 0.807 = fixed MPC). M132: reward weighted like the Phase III score,
# where D is a third of it (carbon 1, ramping 1, peak 1)
METHODS['M132_rl_tuned_mpc_grid'] = dict(METHODS['M129_rl_tuned_mpc'], mu_fixed={'carbon': 1.0, 'ramp': 1.0, 'peak': 1.0},
                                         label='RL-tuned MPC (TQC), reward with equal cost / carbon / grid weights')
METHODS['M131_rl_tuned_gbm_mpc'] = dict(_M27, algo='tqc', env_steps=100_000,
                                        mpc_tune={'horizon': 24, 'w_carbon': 1.0, 'w_grid': 2.0, 'forecaster': 'gbm'},
                                        label='RL-tuned MPC with LightGBM forecasts (TQC)')
# Round 27 (seed 42): improve M131, the best controller (Phase III public 0.782 / private 0.845). One change each
_M131 = METHODS['M131_rl_tuned_gbm_mpc']
METHODS.update({
    'M133_rl_gbm_mpc_score_reward': dict(_M131, mu_fixed={'ramp': 0.5, 'peak': 0.5},
                                         label='RL-tuned LightGBM MPC, score-aligned reward (carbon weight 1)'),
    'M134_rl_gbm_mpc_200k': dict(_M131, env_steps=200_000, label='RL-tuned LightGBM MPC, 200k env steps'),
    'M135_rl_gbm_mpc_sac': dict(_M131, algo='sac', label='RL-tuned LightGBM MPC with SAC'),
})
# Round 29 (seed 42): improve M131 (RL-tuned LightGBM MPC; 3 seeds: Phase III public 0.782, private 0.845)
_M131_tune = METHODS['M131_rl_tuned_gbm_mpc']['mpc_tune']
METHODS.update({
    'M139_rl_mpc_gbm_tuned': dict(_M131, mpc_tune=dict(_M131_tune, forecaster='gbm_tuned'),
                                  label='RL-tuned MPC with Optuna-tuned LightGBM forecasts (TQC)'),
    'M140_rl_mpc_h36': dict(_M131, mpc_tune=dict(_M131_tune, horizon=36), label='RL-tuned LightGBM MPC, 36 h horizon'),
    'M141_rl_mpc_grid3': dict(_M131, mpc_tune=dict(_M131_tune, w_grid=3.0),
                              label='RL-tuned LightGBM MPC, base ramp weight 3'),
    'M142_rl_mpc_gbm_gridreward': dict(_M131, mu_fixed={'carbon': 1.0, 'ramp': 1.0, 'peak': 1.0},
                                       label='RL-tuned LightGBM MPC, reward with equal cost / carbon / grid weights'),
    'M143_rl_mpc_gamma098': dict(_M131, sac_params={'gamma': 0.98}, label='RL-tuned LightGBM MPC, gamma 0.98'),
    'M144_rl_mpc_tuned_h36': dict(_M131, mpc_tune=dict(_M131_tune, forecaster='gbm_tuned', horizon=36),
                                  label='RL-tuned MPC, tuned LightGBM forecasts + 36 h horizon'),
    'M145_rl_mpc_tuned_grid3': dict(_M131, mpc_tune=dict(_M131_tune, forecaster='gbm_tuned', w_grid=3.0),
                                    label='RL-tuned MPC, tuned LightGBM forecasts + base ramp weight 3'),
    'M146_tqc_hist_g098_wear03': dict(METHODS['M123_tqc_hist_gamma098'], w_batt=0.3,
                                      label='TQC + history, gamma 0.98, battery-wear weight 0.3 (pure RL)'),
    'M147_tqc_hist_gamma097': dict(METHODS['M123_tqc_hist_gamma098'], sac_params={'gamma': 0.97},
                                   label='TQC + history, gamma 0.97 (pure RL)'),
})
# Round 30 (seed 42): improve M139 / M144 (RL-tuned MPC with the Optuna-tuned LightGBM forecasts; S 0.777 / 0.778)
_M139_tune = METHODS['M139_rl_mpc_gbm_tuned']['mpc_tune']
METHODS.update({
    'M148_rl_mpc_tuned_h48': dict(_M131, mpc_tune=dict(_M139_tune, horizon=48),
                                  label='RL-tuned MPC, tuned LightGBM forecasts + 48 h horizon'),
    'M149_rl_mpc_tuned_h36_200k': dict(_M131, env_steps=200_000, mpc_tune=dict(_M139_tune, horizon=36),
                                       label='RL-tuned MPC, tuned forecasts + 36 h horizon, 200k env steps'),
    'M150_rl_mpc_tuned_sac': dict(_M131, algo='sac', mpc_tune=_M139_tune,
                                  label='RL-tuned MPC, tuned LightGBM forecasts, SAC'),
    'M151_rl_mpc_tuned_h36_score': dict(_M131, mu_fixed={'ramp': 0.5, 'peak': 0.5},
                                        mpc_tune=dict(_M139_tune, horizon=36),
                                        label='RL-tuned MPC, tuned forecasts + 36 h horizon, carbon weight 1 reward'),
    'M152_rl_mpc_tuned_h30': dict(_M131, mpc_tune=dict(_M139_tune, horizon=30),
                                  label='RL-tuned MPC, tuned LightGBM forecasts + 30 h horizon'),
    'M153_rl_mpc_tuned_h36_carbon15': dict(_M131, mpc_tune=dict(_M139_tune, horizon=36, w_carbon=1.5),
                                           label='RL-tuned MPC, tuned forecasts + 36 h horizon, base carbon weight 1.5'),
    'M154_rl_mpc_tuned_h36_grid15': dict(_M131, mpc_tune=dict(_M139_tune, horizon=36, w_grid=1.5),
                                         label='RL-tuned MPC, tuned forecasts + 36 h horizon, base ramp weight 1.5'),
    'M155_rl_mpc_tuned_h36_sac': dict(_M131, algo='sac', mpc_tune=dict(_M139_tune, horizon=36),
                                      label='RL-tuned MPC, tuned forecasts + 36 h horizon, SAC'),
})
# Round 31 (seed 42): the weak KPI is the load factor (M152: ramping 0.82 but 1 - load factor 0.98, almost no peak
# shaving). The policy gets a 4th action, the MPC's peak-shaving weight (monthly-peak slack, w_peak x 4^a3)
_M152_tune = METHODS['M152_rl_mpc_tuned_h30']['mpc_tune']
METHODS.update({
    'M156_rl_mpc_peak': dict(_M131, mpc_tune=dict(_M152_tune, w_peak=1.0),
                             label='RL-tuned MPC (M152) + RL-set peak-shaving weight'),
    'M157_rl_mpc_peak_reward': dict(_M131, mpc_tune=dict(_M152_tune, w_peak=1.0),
                                    mu_fixed={'carbon': 2.0, 'ramp': 0.5, 'peak': 1.5},
                                    label='RL-tuned MPC (M152) + RL-set peak weight, reward peak weight 1.5'),
    'M158_rl_mpc_peak2': dict(_M131, mpc_tune=dict(_M152_tune, w_peak=2.0),
                              label='RL-tuned MPC (M152) + RL-set peak-shaving weight, base 2'),
})
# Round 32 (seed 42): more RL on top of M152 (3 seeds: Phase III public 0.774, private 0.844). Residual RL: besides the
# MPC weights, the policy adds a district-wide correction to the batteries' MPC action; history: it also sees the past
# 24 h of district net load and the recent forecast error
METHODS.update({
    'M159_rl_mpc_residual': dict(_M131, mpc_tune=dict(_M152_tune, residual=0.25),
                                 label='RL-tuned MPC (M152) + residual action (up to 25% of capacity)'),
    'M160_rl_mpc_residual10': dict(_M131, mpc_tune=dict(_M152_tune, residual=0.10),
                                   label='RL-tuned MPC (M152) + residual action (up to 10% of capacity)'),
    'M161_rl_mpc_hist': dict(_M131, mpc_tune=dict(_M152_tune, obs_history=True),
                             label='RL-tuned MPC (M152) + past 24 h net load and recent forecast error in the observation'),
    'M162_rl_mpc_residual_hist': dict(_M131, mpc_tune=dict(_M152_tune, residual=0.25, obs_history=True),
                                      label='RL-tuned MPC (M152) + residual action + history observation'),
    'M163_rl_mpc_residual_hist_sac': dict(_M131, algo='sac', mpc_tune=dict(_M152_tune, residual=0.25, obs_history=True),
                                          label='RL-tuned MPC (M152) + residual action + history observation, SAC'),
})
# Round 33 (seed 42): bigger actor / critic networks (default 64 x 64) on M152, and one on the best pure RL run (M113)
_M152 = METHODS['M152_rl_mpc_tuned_h30']
METHODS.update({
    'M164_rl_mpc_net256': dict(_M152, net_arch=[256, 256], label='RL-tuned MPC (M152), 256 x 256 networks'),
    'M165_rl_mpc_net512x3': dict(_M152, net_arch=[512, 512, 512], label='RL-tuned MPC (M152), 512 x 512 x 512 networks'),
    'M166_rl_mpc_net1024': dict(_M152, net_arch=[1024, 1024], label='RL-tuned MPC (M152), 1024 x 1024 networks'),
    'M167_rl_mpc_net512x3_200k': dict(_M152, net_arch=[512, 512, 512], env_steps=200_000,
                                      label='RL-tuned MPC (M152), 512 x 512 x 512 networks, 200k env steps'),
    'M168_tqc_hist_net512x3': dict(METHODS['M113_tqc_hist'], net_arch=[512, 512, 512], env_steps=200_000,
                                   label='TQC + history (M113), 512 x 512 x 512 networks, 200k env steps (pure RL)'),
})
# Round 34 (seed 42): M152 with the district-total forecast averaged from LightGBM and the deep F2 MLP
# (colab/F2/F2_district_total.ipynb); the best fixed-MPC forecaster on the unseen buildings
METHODS.update({
    'M169_rl_mpc_totblend': dict(_M152, mpc_tune=dict(_M152_tune, forecaster='gbm_tuned_tot+deep_mlp'),
                                 label='RL-tuned MPC (M152) with the LightGBM + deep MLP district-total forecast'),
    'M170_rl_mpc_totblend_f3': dict(_M152, mpc_tune=dict(_M152_tune, forecaster='gbm_tuned_tot+deep_f3'),
                                    label='RL-tuned MPC (M152) with the LightGBM + F3 district-total forecast'),
})
# Round 28 (seed 42): M131 with the deep forecasters trained on a GPU (colab/F1/F1_deep_forecasters.ipynb, leave-district-out)
METHODS.update({
    'M136_rl_mpc_deep_transformer': dict(_M131, mpc_tune=dict(_M131['mpc_tune'], forecaster='deep_transformer'),
                                         label='RL-tuned MPC with deep Transformer forecasts (TQC)'),
    'M137_rl_mpc_deep_mlp': dict(_M131, mpc_tune=dict(_M131['mpc_tune'], forecaster='deep_mlp'),
                                 label='RL-tuned MPC with deep MLP forecasts (TQC)'),
    'M138_rl_mpc_gbm_mlp_blend': dict(_M131, mpc_tune=dict(_M131['mpc_tune'], forecaster='gbm+deep_mlp'),
                                      label='RL-tuned MPC with LightGBM + deep MLP blended forecasts (TQC)'),
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
        self.history = METHODS[method].get('history_features', False)
        self.hist, self.hist_step = [], None
        low, high = old_obs.low, old_obs.high
        if self.history:
            n, lags = len(self.env.buildings), self.history_lags()
            n_extra = 4 + 2 * len(lags) + ((n * len(lags) + 2 + 4) if self.history == 'rich' else 0)
            low, high = np.append(low, np.full(n_extra, -1e4)), np.append(high, np.full(n_extra, 1e4))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=old_act.low, high=old_act.high, dtype=np.float32)
        # RL-tuned MPC (`mpc_tune`, Round 26): the policy does not output battery actions; each hour it sets the weights of
        # a causal-forecast MPC (ramp weight x 4^a0, carbon weight x 2^a1, end-of-horizon value x (1 + a2/2)), and the
        # MPC turns them into battery actions. Observations are district-level (forecasts of the next 24 h of net load,
        # carbon intensity and price, time, mean SoC, ...), so a policy trained on 5 buildings runs on any district.
        self.mpc_cfg = METHODS[method].get('mpc_tune')
        if self.mpc_cfg:
            c = self.mpc_cfg
            self.ctrl = make_mpc_controller(c, len(self.env.buildings), 'train')
            self.prev_fc = None
            # Round 32: c['obs_history'] adds the past 24 h of district net load and the recent mean |forecast error|
            self.mpc_hist = []
            n_obs = 3 * c['horizon'] + 10 + (25 if c.get('obs_history') else 0)
            self.observation_space = gym.spaces.Box(low=-50.0, high=50.0, shape=(n_obs,), dtype=np.float32)
            # with c['w_peak'] (Round 31) an action sets the peak-shaving weight: w_peak x 4^a; with c['residual']
            # (Round 32) a last action adds residual x a (fraction of capacity) to every battery's MPC action
            n_act = 3 + ('w_peak' in c) + ('residual' in c)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        # Residual methods: base action of each simulated step (row t = action that produced step t), from the RBC cache
        # or, with `residual_base`, from another cached controller's eval trace (e.g. a perfect-foresight LP schedule)
        self.rbc_actions = None
        self.residual_scale = METHODS[method].get('residual_scale', 1.0)
        if METHODS[method].get('residual'):
            self.rbc_actions = load_cached_actions(METHODS[method].get('residual_base', RBC_FOLDER), len(self.env.buildings))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = reset_state(self.state, seed if seed is not None else self.seed_value)
        return self.policy_obs(obs), {}

    def step(self, action):
        obs, reward, done, terms, t = env_step(self.state, self.to_citylearn_actions(action))
        info = {}
        if done:
            info['episode_metrics'] = episode_metrics(self.state)
        return self.policy_obs(obs), reward, done, False, info

    def to_citylearn_actions(self, action):
        if self.mpc_cfg:
            a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            c = self.mpc_cfg
            act = self.ctrl.decide(w_grid=c['w_grid'] * 4.0 ** a[0], w_carbon=c['w_carbon'] * 2.0 ** a[1],
                                   term_frac=1.0 + 0.5 * a[2],
                                   w_peak=c['w_peak'] * 4.0 ** a[3] if 'w_peak' in c else None)
            if 'residual' in c:
                act = np.clip(act + c['residual'] * a[-1], -0.78125, 0.78125).astype(np.float32)
            return [act]
        if self.rbc_actions is None:
            return [action]
        t_next = min(self.env.time_step + 1, len(self.rbc_actions) - 1)   # the step this action will produce
        return [np.clip(self.residual_scale * action + self.rbc_actions[t_next], self.action_space.low, self.action_space.high)]

    def policy_obs(self, citylearn_obs):
        """The policy's observation. With `history_features`, CityLearn's 60 values plus features built only from what
        has already happened (a real controller has them too): cyclic hour / month, and a persistence forecast taken from
        yesterday, i.e. the carbon intensity and the district's non-shiftable load minus solar 24 - h hours ago for the next
        h = 1, 2, 3, 6, 12 hours. Called exactly once per simulated step (training, evaluation), so it keeps the history.
        Round 23 variants: `history_features='dense'` uses every next hour h = 1..12; `'rich'` adds each building's own
        load minus solar at the same lags, the values 24 hours ago, and yesterday's mean / max carbon intensity and
        district load minus solar (all still past-only)."""
        if self.mpc_cfg:
            return self.mpc_obs(citylearn_obs)
        o = np.array(citylearn_obs[0], dtype=np.float32)
        if not self.history:
            return o
        t = self.env.time_step
        if t == 0 or self.hist_step is None or t < self.hist_step:
            self.hist = []
        if self.hist_step == t and self.hist:
            self.hist[-1] = o                                     # same step seen twice: keep one entry
        else:
            self.hist.append(o)
        self.hist_step = t
        n = len(self.env.buildings)
        s, k = CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING
        def ci_and_demand(v):
            demand = sum(v[s + k * i] - v[s + k * i + 1] for i in range(n))   # non_shiftable_load - solar_generation
            return v[19], demand
        back = lambda lag: self.hist[max(len(self.hist) - 1 - lag, 0)]
        lags = self.history_lags()
        lagged = [ci_and_demand(back(lag)) for lag in lags]
        hour, month = o[2], o[0]
        extra = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
        extra += [c for c, _ in lagged] + [d for _, d in lagged]
        if self.history == 'rich':
            extra += [back(lag)[s + k * i] - back(lag)[s + k * i + 1] for i in range(n) for lag in lags]
            extra += list(ci_and_demand(back(24)))
            day = [ci_and_demand(v) for v in self.hist[-24:]]
            cis, demands = [c for c, _ in day], [d for _, d in day]
            extra += [np.mean(cis), np.max(cis), np.mean(demands), np.max(demands)]
        return np.append(o, np.array(extra, dtype=np.float32))

    def mpc_obs(self, citylearn_obs):
        """District-level observation for the RL-tuned MPC (also updates the MPC's forecaster; once per step)."""
        o = np.array(citylearn_obs[0], dtype=np.float64)
        t, B = self.env.time_step, len(self.env.buildings)
        s, k = CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING
        soc = [o[s + k * i + 2] for i in range(B)]
        ctrl = self.ctrl
        ctrl.observe(t, o, soc)
        m = max(abs(ctrl.month_sum / ctrl.month_n), 0.1 * B)     # load scale: this month's mean district load
        err = 0.0 if self.prev_fc is None or t == 0 else (ctrl.base_now - self.prev_fc) / m
        self.prev_fc = ctrl.base_fc[0]
        hour, month = o[2], o[0]
        extra = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * month / 12),
                 np.cos(2 * np.pi * month / 12), float(np.mean(soc)), ctrl.E0 / m, ctrl.base_now / m,
                 ctrl.month_peak / m, err, o[19] / 0.2]
        feats = np.concatenate([ctrl.base_fc / m, ctrl.ci_fc / 0.2, ctrl.price_fc / 0.3, extra])
        if self.mpc_cfg.get('obs_history'):
            if t == 0:
                self.mpc_hist = []
            self.mpc_hist.append((ctrl.base_now / m, abs(err)))
            past = [self.mpc_hist[max(len(self.mpc_hist) - 1 - lag, 0)][0] for lag in range(24)]
            recent_err = float(np.mean([e for _, e in self.mpc_hist[-24:]]))
            feats = np.concatenate([feats, past, [recent_err]])
        return np.clip(feats, -50, 50).astype(np.float32)

    def history_lags(self):
        return HISTORY_LAGS_DENSE if self.history == 'dense' else HISTORY_LAGS


# Lags (hours back) of the persistence forecast: 24 - h for the next h = 1, 2, 3, 6, 12 hours ('dense': h = 1..12)
HISTORY_LAGS = [23, 22, 21, 18, 12]
HISTORY_LAGS_DENSE = list(range(23, 11, -1))


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

class CausalForecaster:
    """Forecasts of the next H hours made only from what has already been observed (deployable in a real building).
    Per series: 0.5 x the same hour yesterday + 0.5 x the mean of that hour over the last 7 days, plus the current
    forecast error (actual - forecast at this hour) fading as 0.8^h. Prices use the same hour one week earlier
    (time-of-use tariffs repeat weekly). Series: each building's non-shiftable load and PV generation, carbon intensity
    and price, read from the central observation."""
    def __init__(self, n_buildings, horizon, max_steps=9000, solar_model=False):
        self.B, self.H = n_buildings, horizon
        self.load = np.zeros((max_steps, n_buildings))
        self.solar = np.zeros((max_steps, n_buildings))
        self.ci = np.zeros(max_steps)
        self.price = np.zeros(max_steps)
        # Solar model (`solar_model=True`): PV of each building ~ a * direct + b * diffuse irradiance, fitted by least
        # squares on past hours only; future irradiance comes from the environment's own 6 / 12 / 24 h weather forecasts
        # (the forecast made at t + h - 6 for h <= 6, at t + h - 12 for h <= 12, at t + h - 24 otherwise)
        self.solar_model = solar_model
        self.irr = np.zeros((max_steps, 2))                  # actual direct, diffuse at t
        self.irr_pred = np.zeros((max_steps, 3, 2))          # forecasts made at t for t+6, t+12, t+24
        self.xtx = np.zeros((n_buildings, 2, 2)); self.xty = np.zeros((n_buildings, 2))

    def update(self, t, o):
        s, k = CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING
        self.load[t] = [o[s + k * i] for i in range(self.B)]
        self.solar[t] = [o[s + k * i + 1] for i in range(self.B)]
        self.ci[t] = o[19]
        self.price[t] = o[s + 4]
        self.irr[t] = [o[15], o[11]]
        self.irr_pred[t] = [[o[16], o[12]], [o[17], o[13]], [o[18], o[14]]]
        if self.solar_model and self.irr[t].sum() > 1.0:
            x = self.irr[t]
            self.xtx += np.outer(x, x)[None]
            self.xty += self.solar[t][:, None] * x[None]

    def _solar_forecast(self, t):
        """(H, B) PV forecast from the weather forecasts already issued, or None before the model has enough data."""
        if self.xtx[0].trace() < 1e3:
            return None
        coef = np.array([np.linalg.solve(self.xtx[b] + 1e-3 * np.eye(2), self.xty[b]) for b in range(self.B)])
        out = np.zeros((self.H, self.B))
        for h in range(1, self.H + 1):
            lead, slot = (6, 0) if h <= 6 else (12, 1) if h <= 12 else (24, 2)
            issued = t + h - lead
            irr = self.irr_pred[issued, slot] if issued >= 0 else self.irr[t]
            out[h - 1] = coef @ irr
        return np.maximum(out, 0.0)

    @staticmethod
    def _seasonal(x, t, h):
        """Same-hour forecast of x[t + h] from x[0..t]: 0.5 x latest same hour + 0.5 x mean of up to 7 same hours."""
        idx = [t + h - 24 * d for d in range(1, 8) if 0 <= t + h - 24 * d <= t]
        if not idx:
            return x[t]
        return 0.5 * x[idx[0]] + 0.5 * x[idx].mean(axis=0)

    def forecast(self, t):
        """(net load per building = load - solar, carbon intensity, price), each for hours t+1 .. t+H."""
        out = {}
        for name, x in [('load', self.load), ('solar', self.solar), ('ci', self.ci)]:
            err = x[t] - self._seasonal(x, t, 0)
            f = np.array([self._seasonal(x, t, h) + err * 0.8 ** h for h in range(1, self.H + 1)])
            out[name] = np.maximum(f, 0.0)
        price = np.array([self.price[t + h - 168] if t + h - 168 >= 0 else self._seasonal(self.price, t, h)
                          for h in range(1, self.H + 1)])
        if self.solar_model:
            model = self._solar_forecast(t)
            if model is not None:
                out['solar'] = model
        return out['load'] - out['solar'], out['ci'], price


class GBMForecaster(CausalForecaster):
    """Load / PV forecasts from gradient-boosted trees (LightGBM), trained offline by tools/forecast_lab.py only on
    buildings OUTSIDE the district being controlled (leave-district-out), as the challenge winners trained on the public
    buildings and were scored on unseen ones. Inputs are available at decision time: recent and lagged values, 7-day
    same-hour means, calendar, current weather and the environment's own weather forecasts for the target hour. Carbon
    intensity and price use the causal seasonal forecasts of CausalForecaster."""
    WEATHER = [3, 7, 11, 15]                         # temperature, humidity, diffuse, direct irradiance (obs indices)
    WEATHER_PRED = [[4, 5, 6], [8, 9, 10], [12, 13, 14], [16, 17, 18]]   # their 6 / 12 / 24 h forecasts

    def __init__(self, n_buildings, horizon, models, max_steps=9000, version=1):
        super().__init__(n_buildings, horizon, max_steps)
        self.models = models                         # {'load': Booster or [Boosters] (seed ensemble), 'solar': ...}
        self.version = version
        self.weather = np.zeros((max_steps, 4))
        self.weather_pred = np.zeros((max_steps, 4, 3))
        self.cal = np.zeros((max_steps, 3))          # month, day type, hour

    def update(self, t, o):
        super().update(t, o)
        self.weather[t] = [o[i] for i in self.WEATHER]
        self.weather_pred[t] = [[o[i] for i in row] for row in self.WEATHER_PRED]
        self.cal[t] = [o[0], o[1], o[2]]

    @staticmethod
    def scale(x, t):
        """Per-building scale: mean |x| over the past week (so models transfer between buildings of different size)."""
        return np.abs(x[max(t - 167, 0):t + 1]).mean(axis=0) + 0.05

    @staticmethod
    def features(x, weather, weather_pred, cal, t, H, version=1):
        """(B * H, n_features) rows for series x (T, B) at time t, horizons 1..H; identical in training and control.
        Every x-derived feature is divided by the building's own scale (the target is x / scale too)."""
        x = x[:t + 1] / GBMForecaster.scale(x, t)
        B = x.shape[1]
        # Vectorised over horizons and buildings (same values and row order, h-major, as the loop version below)
        hs = np.arange(1, H + 1)
        back = lambda lag: x[np.clip(t + hs - lag, 0, t)]                       # (H, B)
        same = np.stack([back(24 * d) for d in range(1, 8)])                    # (7, H, B)
        lead = np.where(hs <= 6, 6, np.where(hs <= 12, 12, 24))
        slot = np.where(hs <= 6, 0, np.where(hs <= 12, 1, 2))
        issued = t + hs - lead
        w_fut = np.where((issued >= 0)[:, None], weather_pred[np.maximum(issued, 0), :, slot], weather[t][None])
        hour = (cal[t, 2] - 1 + hs) % 24 + 1
        daytype = cal[np.clip(t + hs - 168, 0, t), 1]
        recent = x[max(t - 23, 0):t + 1]
        full = lambda v: np.broadcast_to(v, (H, B))
        cols = [full(hs[:, None]), full(hour[:, None]), full(daytype[:, None]), full(cal[t, 0]), full(x[t]),
                full(x[max(t - 1, 0)]), same[0], same[1], back(168), same.mean(axis=0), full(recent.mean(axis=0)),
                full(recent.max(axis=0))]
        cols += [full(weather[t][i]) for i in range(4)] + [full(w_fut[:, i][:, None]) for i in range(4)]
        if version >= 2:
            cols += [full(x[max(t - 2, 0)]), full(x[max(t - 3, 0)]), full(x[t] - x[max(t - 24, 0)]),
                     full(recent.std(axis=0)), full(recent.min(axis=0))]
        return np.stack(cols, axis=-1).reshape(H * B, len(cols)).astype(np.float32)

    @staticmethod
    def features_loop(x, weather, weather_pred, cal, t, H, version=1):
        """Reference (slow) implementation of `features`, kept to check the vectorised one."""
        x = x[:t + 1] / GBMForecaster.scale(x, t)
        B = x.shape[1]
        rows = []
        for h in range(1, H + 1):
            def back(lag):                           # x at t + h - lag, clipped to what is already known
                return x[min(max(t + h - lag, 0), t)]
            same = [back(24 * d) for d in range(1, 8)]
            lead, slot = (6, 0) if h <= 6 else (12, 1) if h <= 12 else (24, 2)
            issued = t + h - lead
            w_fut = weather_pred[issued, :, slot] if issued >= 0 else weather[t]
            hour = (cal[t, 2] - 1 + h) % 24 + 1
            daytype = cal[min(max(t + h - 168, 0), t), 1]
            recent = x[max(t - 23, 0):t + 1]
            for b in range(B):
                row = [h, hour, daytype, cal[t, 0], x[t, b], x[max(t - 1, 0), b], same[0][b], same[1][b],
                       back(168)[b], np.mean([v[b] for v in same]), recent[:, b].mean(), recent[:, b].max(),
                       *weather[t], *w_fut]
                if version >= 2:                 # v2: more recent lags, daily change and variability
                    row += [x[max(t - 2, 0), b], x[max(t - 3, 0), b], x[t, b] - x[max(t - 24, 0), b],
                            recent[:, b].std(), recent[:, b].min()]
                rows.append(row)
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def total_extra(s, temp, t, H):
        """(H, 11) extra features of the district-total load model (tools/load_lab.py, 'direct_x'): s is the scaled
        total load (1-D, known up to t), temp the observed temperature."""
        hs = np.arange(1, H + 1)
        lag = lambda k: s[max(t - k, 0)]
        four_weeks = np.stack([s[np.clip(t + hs - 168 * w, 0, t)] for w in range(1, 5)]).mean(0)
        same_hour_28 = np.stack([s[np.clip(t + hs - 24 * d, 0, t)] for d in range(1, 29)]).mean(0)
        base = [lag(2), lag(3), lag(6), s[max(t - 2, 0):t + 1].mean(), s[max(t - 5, 0):t + 1].mean(),
                s[t] - s[max(t - 24, 0)], s[t] - s[max(t - 168, 0)], s[max(t - 23, 0):t + 1].std(),
                temp[max(t - 23, 0):t + 1].mean()]
        return np.column_stack([np.tile(base, (H, 1)), four_weeks, same_hour_28])

    def forecast_total_load(self, t):
        """District-total load for t+1 .. t+H from the direct total model (self.models['load_total'], LightGBM) and/or
        the deep total models of colab/F2/F2_district_total.ipynb (self.models['load_total_deep']: TorchScript seed
        ensemble, 24 hours; beyond 24 h the LightGBM total is used). With both, their average."""
        x = self.load[:t + 1].sum(1, keepdims=True)
        s = self.scale(x, t)
        totals = []
        if 'load_total' in self.models:
            X = np.hstack([self.features(x, self.weather, self.weather_pred, self.cal, t, self.H),
                           self.total_extra(x[:, 0] / s[0], self.weather[:, 0], t, self.H)])
            totals.append(np.maximum(self.models['load_total'].predict(X), 0.0) * s[0])
        if 'load_total_deep' in self.models:
            import sys as _sys
            colab = str(PHASE_DIR / 'colab' / 'F2')
            if colab not in _sys.path:
                _sys.path.insert(0, colab)
            import total_inputs
            Xd, sd = total_inputs.inputs(x, self.weather, self.weather_pred, self.cal, t)
            with torch.no_grad():
                p = np.mean([m(torch.from_numpy(Xd)).numpy()[0] for m in self.models['load_total_deep']], axis=0)
            deep = np.maximum(p, 0.0) * sd[0]
            if self.H > len(deep):
                assert totals, 'a horizon beyond 24 h needs the LightGBM total model as well'
                deep = np.concatenate([deep, totals[0][len(deep):]])
            totals.append(deep[:self.H])
        return np.mean(totals, axis=0)

    def forecast(self, t):
        _, ci, price = super().forecast(t)
        out = {}
        for name, x in [('load', self.load), ('solar', self.solar)]:
            X = self.features(x, self.weather, self.weather_pred, self.cal, t, self.H, self.version)
            ms = self.models[name] if isinstance(self.models[name], list) else [self.models[name]]
            pred = np.mean([m.predict(X) for m in ms], axis=0).reshape(self.H, self.B) * self.scale(x, t)
            out[name] = np.maximum(pred, 0.0)
        if 'load_total' in self.models or 'load_total_deep' in self.models:
            # reconcile: rescale the per-building load forecasts so that they add up to the direct total forecast
            total, parts = self.forecast_total_load(t), out['load'].sum(1)
            out['load'] = np.where(parts[:, None] > 1e-6, out['load'] * (total / np.maximum(parts, 1e-6))[:, None],
                                   total[:, None] / self.B)
        return out['load'] - out['solar'], ci, price


class DeepForecaster(GBMForecaster):
    """Load / PV forecasts from deep networks (MLP, LSTM or Transformer) trained on a GPU by
    colab/F1/F1_deep_forecasters.ipynb, leave-district-out like the LightGBM ones, exported as TorchScript and run here on
    the CPU. Inputs come from colab/F1/deep_inputs.py, the same file the notebook embeds. `models` is a list of loaded
    TorchScript modules (a seed ensemble, averaged)."""
    def __init__(self, n_buildings, horizon, models, max_steps=9000):
        super().__init__(n_buildings, horizon, None, max_steps)
        import sys as _sys
        colab = str(PHASE_DIR / 'colab' / 'F1')
        if colab not in _sys.path:
            _sys.path.insert(0, colab)
        import deep_inputs
        self.di, self.nets = deep_inputs, models

    def forecast(self, t):
        _, ci, price = CausalForecaster.forecast(self, t)
        X, sl, ss = self.di.inputs(self.load, self.solar, self.weather, self.weather_pred, self.cal, t)
        with torch.no_grad():
            Xt = torch.tensor(X)
            P = np.mean([m(Xt).numpy() for m in self.nets], axis=0)       # (B, 2H)
        H = self.H
        load = np.maximum(P[:, :H] * sl[:, None], 0.0).T                   # (H, B)
        solar = np.maximum(P[:, H:2 * H] * ss[:, None], 0.0).T
        return load - solar, ci, price


def load_deep_models(district, arch):
    """TorchScript seed ensemble from colab/F1/deep_forecasters/ (unzipped from F1_deep_forecasters_results.zip)."""
    folder = PHASE_DIR / 'colab' / 'F1' / 'deep_forecasters'
    paths = sorted(folder.glob(f'{district}_{arch}_s*.pt'))
    assert paths, f'no {district}_{arch} models in {folder}'
    return [torch.jit.load(str(p), map_location='cpu').eval() for p in paths]


class BlendForecaster(CausalForecaster):
    """Online ensemble of the causal seasonal forecaster and GBMForecaster, as CUFE's meta-learner and Together's
    self-adaptive adjustment: per building and horizon, each forecast is weighted by the inverse of its recent absolute
    error (exponential average over forecasts that have since come true), then the latest observed error of the blend
    is added, fading as 0.8^h. Everything is computed from the past only."""
    def __init__(self, n_buildings, horizon, models, max_steps=9000, memory=0.98):
        super().__init__(n_buildings, horizon, max_steps)
        self.gbm = GBMForecaster(n_buildings, horizon, models, max_steps)
        self.memory = memory
        self.store = {}                              # t -> (seasonal (H, B), gbm (H, B), blend (H, B)) net forecasts
        self.err = np.ones((2, horizon, n_buildings))

    def update(self, t, o):
        super().update(t, o)
        self.gbm.update(t, o)

    def forecast(self, t):
        seas_net, ci, price = CausalForecaster.forecast(self, t)
        seas_net = self._per_building(t)
        gbm_net, _, _ = self.gbm.forecast(t)
        actual = self.load[t] - self.solar[t]
        for h in range(1, self.H + 1):               # score the forecasts made h hours ago for this hour
            past = self.store.get(t - h)
            if past is not None:
                for m in range(2):
                    self.err[m, h - 1] = self.memory * self.err[m, h - 1] + (1 - self.memory) * np.abs(past[m][h - 1] - actual)
        inv = 1.0 / np.maximum(self.err, 1e-3)
        blend = (inv[0] * seas_net + inv[1] * gbm_net) / (inv[0] + inv[1])
        last = self.store.get(t - 1)
        if last is not None:                          # self-adaptive correction with the newest error
            e = actual - last[2][0]
            blend = blend + e[None] * 0.8 ** np.arange(1, self.H + 1)[:, None]
        self.store[t] = (seas_net, gbm_net, blend)
        self.store.pop(t - self.H - 1, None)
        return blend, ci, price

    def _per_building(self, t):
        """Seasonal net forecast per building (H, B), as CausalForecaster.forecast but without its own correction
        being lost to the district sum."""
        out = {}
        for name, x in [('load', self.load), ('solar', self.solar)]:
            err = x[t] - self._seasonal(x, t, 0)
            out[name] = np.maximum(np.array([self._seasonal(x, t, h) + err * 0.8 ** h for h in range(1, self.H + 1)]), 0)
        return out['load'] - out['solar']


class BatteryMPC:
    """Receding-horizon linear program for the district's batteries, rebuilt each hour from causal forecasts.
    Minimises sum_h [price_h / c + w_carbon * ci_h / g] * max(0, E_h)  (+ w_grid/2 * |E_h - E_{h-1}| / r)
    - term_value * stored energy at the horizon end, with the battery model of tools/lp_oracle.py (capacity 6.4 kWh,
    5 kW, sqrt(0.9) efficiency each way, power taper above 80% SoC). c, g, r are running no-battery means, so cost and
    carbon are weighted like the KPI ratios. Only the first hour of the plan is applied."""
    CAP, PMAX, ETA = 6.4, 5.0, 0.9 ** 0.5

    def __init__(self, n_buildings, horizon, w_carbon=1.0, w_grid=0.0, term_frac=1.0, w_peak=0.0, eta=None):
        import scipy.sparse as sp
        if eta is not None:                  # one-way efficiency of the LP's battery model (default sqrt(0.9))
            self.ETA = eta
        self.B, self.H, self.w_carbon, self.w_grid, self.term_frac = n_buildings, horizon, w_carbon, w_grid, term_frac
        self.w_peak = w_peak
        self.has_ramp, self.has_peak = w_grid > 0, w_peak > 0   # LP structure is fixed; weights may change per call
        B, H = n_buildings, horizon
        self.ich = np.arange(B * H).reshape(B, H)
        self.idis = self.ich + B * H
        self.isoc = self.ich + 2 * B * H
        self.iimp = 3 * B * H + np.arange(H)
        self.irp = 3 * B * H + H + np.arange(H)
        self.ipk = 3 * B * H + 2 * H              # slack: district load above the running monthly peak (CUFE-style)
        self.nv = 3 * B * H + 2 * H + 1
        eq_r, eq_c, eq_v = [], [], []
        for b in range(B):
            for h in range(H):
                q = b * H + h
                eq_r += [q, q, q]; eq_c += [self.isoc[b, h], self.ich[b, h], self.idis[b, h]]
                eq_v += [1.0, -self.ETA, 1.0 / self.ETA]
                if h > 0:
                    eq_r.append(q); eq_c.append(self.isoc[b, h - 1]); eq_v.append(-1.0)
        self.A_eq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(B * H, self.nv))
        r_, c_, v_ = [], [], []
        row = 0
        def net(h, sign):            # sign * sum_b (ch - dis) at hour h
            return [(self.ich[b, h], sign) for b in range(B)] + [(self.idis[b, h], -sign) for b in range(B)]
        for h in range(H):           # imports: net_h - imp_h <= -base_h
            for col, v in net(h, 1.0) + [(self.iimp[h], -1.0)]:
                r_.append(row); c_.append(col); v_.append(v)
            row += 1
        self.n_imp_rows = H
        if w_grid > 0:               # ramping: +-(E_h - E_{h-1}) <= rp_h
            for sign in (1.0, -1.0):
                for h in range(H):
                    entries = net(h, sign) + (net(h - 1, -sign) if h > 0 else []) + [(self.irp[h], -1.0)]
                    for col, v in entries:
                        r_.append(row); c_.append(col); v_.append(v)
                    row += 1
        if w_peak > 0:               # peak: net_h - pk <= M - base_h
            for h in range(H):
                for col, v in net(h, 1.0) + [(self.ipk, -1.0)]:
                    r_.append(row); c_.append(col); v_.append(v)
                row += 1
        self.taper_start = row       # power taper: ch, dis <= 4.2 PMAX - 4 PMAX / CAP * soc_prev
        for arr in (self.ich, self.idis):
            for b in range(B):
                for h in range(H):
                    r_.append(row); c_.append(arr[b, h]); v_.append(1.0)
                    if h > 0:
                        r_.append(row); c_.append(self.isoc[b, h - 1]); v_.append(4 * self.PMAX / self.CAP)
                    row += 1
        self.A_ub = sp.csr_matrix((v_, (r_, c_)), shape=(row, self.nv))
        self.bounds = np.zeros((self.nv, 2)); self.bounds[:, 1] = np.inf
        self.bounds[self.ich.ravel(), 1] = self.PMAX; self.bounds[self.idis.ravel(), 1] = self.PMAX
        self.bounds[self.isoc.ravel(), 1] = self.CAP

    def plan(self, base, ci, price, soc0, E0, scales, peak_ref=0.0, mean_load=1.0):
        """Target state of charge (kWh) of each battery after the next hour."""
        from scipy.optimize import linprog
        cost, b_ub, b_eq = self.lp_pieces(base, ci, price, soc0, E0, scales, peak_ref, mean_load)
        res = linprog(cost, A_ub=self.A_ub, b_ub=b_ub, A_eq=self.A_eq, b_eq=b_eq, bounds=self.bounds, method='highs')
        if res.status != 0:
            return soc0
        return res.x[self.isoc[:, 0]]

    def lp_pieces(self, base, ci, price, soc0, E0, scales, peak_ref=0.0, mean_load=1.0):
        """Objective and right-hand sides of the LP for one forecast of the district base load."""
        B, H = self.B, self.H
        c_s, g_s, r_s = scales
        u = price / c_s + self.w_carbon * ci / g_s
        cost = np.zeros(self.nv)
        cost[self.ich.ravel()] = 1e-6; cost[self.idis.ravel()] = 1e-6       # no simultaneous charge and discharge
        cost[self.iimp] = u
        cost[self.isoc[:, -1]] = -self.term_frac * self.ETA * np.median(u)  # value of energy left at the horizon end
        b_ub = [-base]
        if self.has_ramp:
            cost[self.irp] = self.w_grid / 2 / r_s
            d = np.diff(np.concatenate([[E0], base]))
            b_ub += [-d, d]
        if self.has_peak:
            # each kWh above the month's running peak lowers the load factor by about 1 / peak; scaled like the KPI
            cost[self.ipk] = self.w_peak / max(mean_load, 1e-3)
            b_ub += [peak_ref - base]
        taper = np.full((B, H), 4.2 * self.PMAX)
        taper[:, 0] -= 4 * self.PMAX / self.CAP * soc0
        b_ub += [taper.ravel(), taper.ravel()]
        b_eq = np.zeros(B * H); b_eq[np.arange(B) * H] = soc0
        return cost, np.concatenate(b_ub), b_eq


class ScenarioMPC(BatteryMPC):
    """Two-stage stochastic version of BatteryMPC: K forecasts of the district base load (the point forecast plus K-1
    past forecast errors), one battery plan per scenario, the first hour's charge / discharge shared by all of them
    (non-anticipativity), and the mean objective over the scenarios. Only the shared first hour is applied, so the
    action hedges against the forecast errors instead of trusting the point forecast."""

    def __init__(self, n_buildings, horizon, w_carbon=1.0, w_grid=0.0, term_frac=1.0, w_peak=0.0, scenarios=8):
        import scipy.sparse as sp
        super().__init__(n_buildings, horizon, w_carbon, w_grid, term_frac, w_peak)
        K, B, n = scenarios, n_buildings, self.nv
        self.K = K
        self.A_ub_K = sp.block_diag([self.A_ub] * K, format='csr')
        na_r, na_c, na_v = [], [], []                 # ch / dis of hour 0 equal to scenario 0's
        row = 0
        for k in range(1, K):
            for arr in (self.ich, self.idis):
                for b in range(B):
                    na_r += [row, row]; na_c += [k * n + arr[b, 0], arr[b, 0]]; na_v += [1.0, -1.0]
                    row += 1
        self.n_na = row
        self.A_eq_K = sp.vstack([sp.block_diag([self.A_eq] * K, format='csr'),
                                 sp.csr_matrix((na_v, (na_r, na_c)), shape=(row, K * n))], format='csr')
        self.bounds_K = np.tile(self.bounds, (K, 1))

    def plan_scenarios(self, bases, ci, price, soc0, E0, scales, peak_ref=0.0, mean_load=1.0):
        """bases: (K, H) district base-load scenarios (row 0 = the point forecast). Target SoC after the next hour."""
        from scipy.optimize import linprog
        parts = [self.lp_pieces(b, ci, price, soc0, E0, scales, peak_ref, mean_load) for b in bases]
        cost = np.concatenate([p[0] for p in parts]) / len(bases)
        b_ub = np.concatenate([p[1] for p in parts])
        b_eq = np.concatenate([p[2] for p in parts] + [np.zeros(self.n_na)])
        res = linprog(cost, A_ub=self.A_ub_K, b_ub=b_ub, A_eq=self.A_eq_K, b_eq=b_eq, bounds=self.bounds_K,
                      method='highs')
        if res.status != 0:
            return self.plan(bases[0], ci, price, soc0, E0, scales, peak_ref, mean_load)
        return res.x[self.isoc[:, 0]]


class ErrorBank:
    """Past out-of-sample forecast errors of the district net load (tools/error_bank.py), relative to the past-week
    mean district load; scenarios for hour of day `hour` are drawn from errors of forecasts made at that hour."""

    def __init__(self, district, k, seed=0):
        d = np.load(PHASE_DIR / 'results' / 'forecast' / 'error_bank' / f'{district}.npz')
        self.errors, self.hours, self.k = d['errors'], d['hours'], k
        self.by_hour = {h: np.flatnonzero(self.hours == h) for h in range(1, 25)}
        self.rng = np.random.default_rng(seed)

    def scenarios(self, point, hour, scale):
        """(k, H) scenarios: the point forecast, then the point forecast plus k-1 sampled past errors."""
        idx = self.rng.choice(self.by_hour[int(hour)], self.k - 1, replace=False)
        return np.vstack([point, point + self.errors[idx, :len(point)] * scale])


class AverageForecaster:
    """Equal-weight average of several forecasters' net-load forecasts (carbon intensity and price from the first).
    Each member keeps its own history; update / forecast are forwarded to all of them."""
    def __init__(self, members):
        self.members = members

    def update(self, t, o):
        for m in self.members:
            m.update(t, o)

    def forecast(self, t):
        outs = [m.forecast(t) for m in self.members]
        return np.mean([o[0] for o in outs], axis=0), outs[0][1], outs[0][2]


def make_mpc_controller(cfg, n_buildings, district):
    """MPCController for an `mpc_tune` config; with cfg['forecaster'] == 'gbm' it uses the LightGBM forecasters of
    tools/forecast_lab.py for that district (trained only on the other districts' buildings)."""
    fc = None
    fname = cfg.get('forecaster', '')
    if fname in ('gbm', 'gbm_tuned', 'gbm_tuned_tot') or fname.startswith(('gbm_tuned_totdeep_', 'gbm_tuned_tot+deep_')):
        # gbm_tuned: the Optuna-tuned models of tools/tune_forecaster.py (nested leave-district-out); a series not tuned
        # yet for this district falls back to the default model
        # gbm_tuned_tot: as gbm_tuned, with the per-building load forecasts rescaled to the direct district-total
        # model of tools/load_lab.py (trained on pseudo-districts of the other districts' buildings)
        # gbm_tuned_totdeep_<arch>: as gbm_tuned_tot with the deep total model of colab/F2/F2_district_total.ipynb instead
        # gbm_tuned_tot+deep_<arch>: the average of the LightGBM and the deep total models
        import lightgbm as lgb
        base, tuned = PHASE_DIR / 'results' / 'forecast', PHASE_DIR / 'results' / 'forecast' / 'tuned'
        models = {}
        for n in ['load', 'solar']:
            path = tuned / f'{district}_{n}.txt'
            if fname == 'gbm' or not path.exists():
                path = base / f'{district}_{n}.txt'
            models[n] = lgb.Booster(model_file=str(path))
        # (with the deep model alone, the LightGBM total covers only the hours beyond its 24)
        if fname == 'gbm_tuned_tot' or fname.startswith('gbm_tuned_tot+deep_') or \
                (fname.startswith('gbm_tuned_totdeep_') and cfg['horizon'] > 24):
            models['load_total'] = lgb.Booster(model_file=str(base / 'load_lab' / f'{district}_total_direct_x.txt'))
        if 'deep_' in fname:
            # <arch> may join several with '+' (e.g. 'mlp+f3'): one deep ensemble of all their seeds; 'f3' is the
            # Optuna-searched model of colab/F3/F3_automl_total.ipynb (same inputs as F2)
            paths = []
            for arch in fname.split('deep_')[1].split('+'):
                folder = PHASE_DIR / 'colab' / ('F3' if arch == 'f3' else 'F2') / 'total_forecasters'
                found = sorted(folder.glob(f"{district}_{arch}_s*.pt"))
                assert found, f'no {district} {arch} total models in {folder}'
                paths += found
            models['load_total_deep'] = [torch.jit.load(str(p), map_location='cpu').eval() for p in paths]
        fc = GBMForecaster(n_buildings, cfg['horizon'], models)
    elif cfg.get('forecaster', '').startswith('deep_'):             # e.g. 'deep_mlp', 'deep_lstm', 'deep_transformer'
        fc = DeepForecaster(n_buildings, cfg['horizon'], load_deep_models(district, cfg['forecaster'][5:]))
    elif cfg.get('forecaster', '').startswith('gbm+deep_'):         # e.g. 'gbm+deep_mlp': 50/50 average of the two
        import lightgbm as lgb
        folder = PHASE_DIR / 'results' / 'forecast'
        models = {n: lgb.Booster(model_file=str(folder / f'{district}_{n}.txt')) for n in ['load', 'solar']}
        fc = AverageForecaster([GBMForecaster(n_buildings, cfg['horizon'], models),
                                DeepForecaster(n_buildings, cfg['horizon'],
                                               load_deep_models(district, cfg['forecaster'].split('deep_')[1]))])
    # cfg['scenarios'] = K > 1: scenario MPC with K scenarios from the district's forecast-error bank
    bank = ErrorBank(district, cfg['scenarios']) if cfg.get('scenarios', 0) > 1 else None
    return MPCController(n_buildings, cfg['horizon'], cfg['w_carbon'], cfg['w_grid'], 1.0, cfg.get('w_peak', 0.0),
                         forecaster=fc, bank=bank, exact_eff=cfg.get('exact_eff', False), eta=cfg.get('eta'))


# CityLearn 2022 battery (identical in every building): efficiency vs |power| / nominal power (5 kW), linear in between;
# the stored energy changes by grid energy x sqrt(efficiency) when charging, / sqrt(efficiency) when discharging
EFF_CURVE = (np.array([0.0, 0.3, 0.7, 0.8, 1.0]), np.array([0.83, 0.83, 0.9, 0.9, 0.85]))


def battery_energy_for(delta, nominal_power=5.0):
    """Grid-side energy (kWh, per battery) that changes the stored energy by `delta` kWh in CityLearn: solves
    e * sqrt(eff(e)) = delta (charging) or e / sqrt(eff(e)) = delta (discharging) by fixed-point iteration."""
    delta = np.asarray(delta, dtype=np.float64)
    e = np.where(delta >= 0, delta / 0.9 ** 0.5, delta * 0.9 ** 0.5)
    for _ in range(8):
        root = np.sqrt(np.interp(np.abs(e) / nominal_power, *EFF_CURVE))
        e = np.where(delta >= 0, delta / root, delta * root)
    return e


class MPCController:
    """Causal forecaster + BatteryMPC, driven by the central observation once per simulated step. `act` returns the
    CityLearn action (fraction of capacity, grid side) that moves each battery to the MPC's target state of charge."""
    def __init__(self, n_buildings, horizon=24, w_carbon=1.0, w_grid=0.0, term_frac=1.0, w_peak=0.0, solar_model=False,
                 forecaster=None, bank=None, exact_eff=False, eta=None):
        # bank: an ErrorBank -> scenario MPC (ScenarioMPC with bank.k scenarios); None -> deterministic BatteryMPC
        # exact_eff (Round 33): convert the target SoC into an action with CityLearn's power-dependent efficiency
        # curve instead of the LP's constant one; eta: the LP's one-way efficiency
        self.fc = forecaster if forecaster is not None else CausalForecaster(n_buildings, horizon, solar_model=solar_model)
        self.bank, self.exact_eff = bank, exact_eff
        self.mpc = BatteryMPC(n_buildings, horizon, w_carbon, w_grid, term_frac, w_peak, eta) if bank is None else             ScenarioMPC(n_buildings, horizon, w_carbon, w_grid, term_frac, w_peak, scenarios=bank.k)
        self.B = n_buildings
        self.sums = np.zeros(3); self.n = 0; self.prev_base = None
        self.month_peak, self.month_sum, self.month_n = 0.0, 0.0, 0

    def act(self, t, o, soc_frac):
        self.observe(t, o, soc_frac)
        return self.decide()

    def observe(self, t, o, soc_frac):
        """Updates the forecaster and running statistics with the observation of hour t, and forecasts the horizon."""
        s, k = CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING
        if t == 0:
            self.sums[:] = 0; self.n = 0; self.prev_base = None
        if t % 730 == 0:                           # the KPI's load factor uses 730-hour months
            self.month_peak, self.month_sum, self.month_n = 0.0, 0.0, 0
        E_now = sum(o[s + k * i + 3] for i in range(self.B))
        self.month_peak = max(self.month_peak, E_now); self.month_sum += E_now; self.month_n += 1
        self.fc.update(t, o)
        base_now = sum(o[s + k * i] - o[s + k * i + 1] for i in range(self.B))
        self.sums += [max(0.0, base_now * o[s + 4]), max(0.0, base_now * o[19]),
                      abs(base_now - self.prev_base) if self.prev_base is not None else 0.0]
        self.n += 1; self.prev_base = base_now
        self.scales = np.maximum(self.sums / self.n, 1e-3)
        base, self.ci_fc, self.price_fc = self.fc.forecast(t)
        self.base_fc = base.sum(axis=1)
        self.E0, self.base_now = E_now, base_now
        self.soc0 = np.asarray(soc_frac, dtype=np.float64) * BatteryMPC.CAP
        if self.bank is not None:
            self.hour = o[2]
            self.load_scale = self.fc.load[max(t - 167, 0):t + 1].sum(axis=1).mean() + 0.05

    def decide(self, w_grid=None, w_carbon=None, term_frac=None, w_peak=None):
        """CityLearn actions from the MPC plan; the weights default to the controller's own (RL can override them)."""
        m = self.mpc
        saved = (m.w_grid, m.w_carbon, m.term_frac, m.w_peak)
        m.w_grid = saved[0] if w_grid is None else w_grid
        m.w_carbon = saved[1] if w_carbon is None else w_carbon
        m.term_frac = saved[2] if term_frac is None else term_frac
        m.w_peak = saved[3] if w_peak is None else w_peak
        soc0 = self.soc0
        if self.bank is None:
            target = m.plan(self.base_fc, self.ci_fc, self.price_fc, soc0, self.E0, self.scales,
                            peak_ref=self.month_peak, mean_load=self.month_sum / self.month_n)
        else:
            bases = self.bank.scenarios(self.base_fc, self.hour, self.load_scale)
            target = m.plan_scenarios(bases, self.ci_fc, self.price_fc, soc0, self.E0, self.scales,
                                      peak_ref=self.month_peak, mean_load=self.month_sum / self.month_n)
        m.w_grid, m.w_carbon, m.term_frac, m.w_peak = saved
        delta = target - soc0
        if self.exact_eff:
            energy = battery_energy_for(delta)
        else:
            energy = np.where(delta >= 0, delta / m.ETA, delta * m.ETA)
        return np.clip(energy / BatteryMPC.CAP, -0.78125, 0.78125).astype(np.float32)


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
    if isinstance(model, OffPolicyAlgorithm):
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


# Off-policy algorithms (Round 22): SAC, TQC (truncated distributional critics, sb3-contrib) and CrossQ (batch-normalised
# critics without target networks, sb3-contrib). CrossQ has no target network, so no `tau`; its paper uses lr 1e-3.
OFF_POLICY_ALGOS = {'sac': SAC, 'tqc': TQC, 'crossq': CrossQ}

def is_off_policy(method):
    return METHODS[method].get('algo') in OFF_POLICY_ALGOS

def algo_class(method):
    return OFF_POLICY_ALGOS.get(METHODS[method].get('algo'), PPO)

def sac_settings(method):
    """Off-policy hyperparameters and policy kwargs, with the method's optional `sac_params` / `net_arch` overrides."""
    cfg = METHODS[method]
    params = dict(SAC_PARAMS)
    if cfg.get('algo') == 'crossq':
        params.pop('tau')
        params['learning_rate'] = 1e-3
    params.update(cfg.get('sac_params', {}))
    if cfg.get('lr_decay'):
        # Linear decay to 0 over training (SB3 schedules take progress_remaining, 1 at the start and 0 at the end), so the
        # policy settles instead of drifting once it has converged (M109: constant-lr training to 400k got worse)
        params['learning_rate'] = get_linear_fn(params['learning_rate'], 0.0, 1.0)
    return params, {'net_arch': cfg.get('net_arch', SAC_POLICY_KWARGS['net_arch']), **cfg.get('policy_kwargs', {})}


def make_training_env(method, base, run_dir):
    """The env SB3 trains on: the base env, wrapped in VecNormalize (restored from the run's stats) if requested."""
    if not METHODS[method].get('normalize'):
        return base
    venv = base if isinstance(base, VecEnv) else DummyVecEnv([lambda: base])
    if (run_dir / 'vecnormalize.pkl').exists():
        return VecNormalize.load(run_dir / 'vecnormalize.pkl', venv)
    gamma = sac_settings(method)[0]['gamma'] if is_off_policy(method) else PPO_PARAMS['gamma']
    return VecNormalize(venv, norm_obs=True, norm_reward=METHODS[method].get('normalize_reward', True), clip_obs=10.0,
                        gamma=gamma)


BC_EPOCHS = 30 if SMOKE_TEST else 300

CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING = 20, 8   # central observation: 20 shared values, then 8 per building

def permute_buildings(X_raw, Y_unit, n_perm, rng):
    """Building-order augmentation for cloning: n_perm copies of the data with the 5 per-building observation blocks and
    the 5 actions shuffled the same way. The LP plan treats buildings symmetrically, so a shuffled district's optimal
    plan is the shuffled plan: free extra data that stops the clone from tying a strategy to a building slot."""
    n = Y_unit.shape[1]
    s, k = CENTRAL_SHARED_OBS, CENTRAL_OBS_PER_BUILDING
    assert X_raw.shape[1] == s + k * n, 'unexpected central observation layout'
    xs, ys = [X_raw], [Y_unit]
    for _ in range(n_perm):
        p = rng.permutation(n)
        Xp = X_raw.copy()
        for new, old in enumerate(p):
            Xp[:, s + k * new: s + k * (new + 1)] = X_raw[:, s + k * old: s + k * (old + 1)]
        xs.append(Xp)
        ys.append(Y_unit[:, p])
    return np.concatenate(xs), np.concatenate(ys)


def save_bc_state(run_dir, model, opt, env, X, Y, iters_done):
    """Resumable behaviour-cloning state (data, actor, optimiser, observation statistics); bc_state.json written last."""
    rms = env.obs_rms if isinstance(env, VecNormalize) else None
    np.savez(run_dir / 'bc_data_tmp.npz', X=X, Y=Y,
             **({'mean': rms.mean, 'var': rms.var, 'count': np.array(rms.count)} if rms is not None else {}))
    os.replace(run_dir / 'bc_data_tmp.npz', run_dir / 'bc_data.npz')
    torch.save({'actor': model.actor.state_dict(), 'opt': opt.state_dict()}, run_dir / 'bc_actor_tmp.pt')
    os.replace(run_dir / 'bc_actor_tmp.pt', run_dir / 'bc_actor.pt')
    with open(run_dir / 'bc_state_tmp.json', 'w') as f:
        json.dump({'dagger_iters_done': iters_done}, f)
    os.replace(run_dir / 'bc_state_tmp.json', run_dir / 'bc_state.json')


def bc_pretrain(model, method, base, env, run_dir=None):
    """Behaviour cloning: fits the SAC actor's deterministic action to a cached controller's actions (`bc_from`, e.g. a
    perfect-foresight LP plan), on the observations seen along that controller's own trajectory. The cloned policy only
    needs the current observation, not the plan. With VecNormalize, the observation statistics are set from that data.
    With `run_dir`, the state is saved after the initial fit and after every DAgger iteration, and a restarted run
    continues from the last saved iteration (a power cut no longer loses the whole cloning phase)."""
    plan = load_cached_actions(METHODS[method]['bc_from'], len(base.env.buildings))
    scale = env.normalize_obs if isinstance(env, VecNormalize) else (lambda o: o)
    to_unit = lambda a: np.clip(model.policy.scale_action(np.asarray(a)), -0.999, 0.999)   # actor's [-1, 1] range
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    n_perm = METHODS[method].get('bc_permute', 0)
    rng = np.random.default_rng(base.seed_value)
    resume = run_dir is not None and (run_dir / 'bc_state.json').exists()
    if resume:
        with open(run_dir / 'bc_state.json') as f:
            start_iter = json.load(f)['dagger_iters_done']
        # np.load of an .npz keeps the file open until closed, and Windows then refuses to replace it at the next save
        with np.load(run_dir / 'bc_data.npz') as data:
            X, Y = np.array(data['X']), np.array(data['Y'])
            rms = (np.array(data['mean']), np.array(data['var']), float(data['count'])) if 'mean' in data else None
        if isinstance(env, VecNormalize) and rms is not None:
            env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count = rms
        saved = torch.load(run_dir / 'bc_actor.pt')
        model.actor.load_state_dict(saved['actor'])
        opt.load_state_dict(saved['opt'])
        print(f"  behaviour cloning resumed after {start_iter} DAgger iteration(s), {len(X):,} samples")
    else:
        start_iter = 0
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
        X_raw, Y_unit = np.array(obs_list, dtype=np.float32), to_unit(act_list)
        if n_perm:
            X_raw, Y_unit = permute_buildings(X_raw, Y_unit, n_perm, rng)
        if isinstance(env, VecNormalize):
            env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count = X_raw.mean(0), X_raw.var(0) + 1e-8, float(len(X_raw))
        X, Y = scale(X_raw), Y_unit

    noise = METHODS[method].get('bc_obs_noise', 0.0)   # Gaussian input noise (normalised units) for robustness

    def fit(X, Y, epochs):
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        for epoch in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                idx = perm[i:i + 256]
                xb = Xt[idx] + noise * torch.randn_like(Xt[idx]) if noise > 0 else Xt[idx]
                loss = torch.nn.functional.mse_loss(model.actor(xb, deterministic=True), Yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return torch.nn.functional.mse_loss(model.actor(Xt, deterministic=True), Yt).item()

    if not resume:
        final = fit(X, Y, BC_EPOCHS)
        print(f"  behaviour cloning from {METHODS[method]['bc_from']}: {len(X):,} samples, {BC_EPOCHS} epochs, "
              f"final MSE {final:.4f} (actions scaled to [-1, 1])")
        if run_dir is not None:
            save_bc_state(run_dir, model, opt, env, X, Y, 0)
    # DAgger: roll the clone out, label every state it visits with the expert's action there, aggregate, refit. The expert
    # is the plan's action for that hour plus a correction of `bc_dagger_gain` x the state-of-charge error (gain 1 = catch
    # up in one hour, which Round 8 showed makes the clone jumpy; smaller gains recover over several hours)
    plan_soc = load_cached_actions(METHODS[method]['bc_from'], len(base.env.buildings), prefix='soc_b')
    gain = METHODS[method].get('bc_dagger_gain', 1.0)
    eta = 0.9 ** 0.5
    for it in range(start_iter, METHODS[method].get('bc_dagger_iters', 0)):
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
        new_X_raw, new_Y_unit = np.array(new_X, dtype=np.float32), to_unit(new_Y)
        if n_perm:
            new_X_raw, new_Y_unit = permute_buildings(new_X_raw, new_Y_unit, n_perm, rng)
        X, Y = np.concatenate([X, scale(new_X_raw)]), np.concatenate([Y, new_Y_unit])
        final = fit(X, Y, max(BC_EPOCHS // 3, 1))
        print(f"  DAgger iteration {it + 1}: {len(X):,} samples, MSE {final:.4f}")
        if run_dir is not None:
            save_bc_state(run_dir, model, opt, env, X, Y, it + 1)


def train_run(method, seed, rbc):
    """Trains (or resumes, or loads) one method x seed. Returns (model, base env, training env)."""
    run_dir = run_dir_of(method, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = METHODS[method]
    algo = algo_class(method)
    base = make_env(method, seed, rbc)
    env = make_training_env(method, base, run_dir)
    n_envs = base.num_envs if isinstance(base, VecEnv) else 1

    if (run_dir / 'model.zip').exists():
        print("  training already finished - loading model.zip")
        model = algo.load(run_dir / 'model.zip', env=env, device='cpu')
        with open(run_dir / 'checkpoint.json') as f:
            base.state['mu'] = {k: np.array(v) for k, v in json.load(f)['mu'].items()}
        return model, base, env

    if is_off_policy(method):
        sac_params, sac_policy_kwargs = sac_settings(method)
        new_model = lambda: algo('MlpPolicy', env, policy_kwargs=sac_policy_kwargs, seed=seed, device='cpu', **sac_params)
    else:
        # Per-building: 5 samples per env step, so batch size x5 keeps the number of gradient steps equal to central
        params = dict(PPO_PARAMS, batch_size=PPO_PARAMS['batch_size'] * n_envs)
        new_model = lambda: PPO('MlpPolicy', env, policy_kwargs=POLICY_KWARGS, seed=seed, device='cpu', **params)

    if (run_dir / 'checkpoint.zip').exists():
        with open(run_dir / 'checkpoint.json') as f:
            ckpt = json.load(f)
        model = algo.load(run_dir / 'checkpoint.zip', env=env, device='cpu')
        if is_off_policy(method) and (run_dir / 'replay_buffer.pkl').exists():
            model.load_replay_buffer(run_dir / 'replay_buffer.pkl')
        base.state['mu'] = {k: np.array(v) for k, v in ckpt['mu'].items()}
        log_rows = ckpt['log']
        print(f"  resuming from checkpoint at env step {ckpt['env_steps']:,}")
    else:
        model = new_model()
        log_rows = []
        if cfg.get('bc_from'):
            bc_pretrain(model, method, base, env, run_dir)

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


def pid_alive(pid):
    """True if a process with this id is running (Windows: OpenProcess + GetExitCodeProcess; elsewhere os.kill(pid, 0))."""
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259                                               # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def claim_seed(run_dir):
    """Claims a seed folder for this process (claim.json holds its pid). False if a live other process holds it, so
    several processes (e.g. one notebook per seed) can work on one method without two writers on the same seed."""
    claim = run_dir / 'claim.json'
    if claim.exists():
        try:
            pid = json.load(open(claim))['pid']
        except (OSError, ValueError, KeyError):
            pid = None
        if pid is not None and pid != os.getpid() and pid_alive(pid):
            return False
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(claim, 'w') as f:
        json.dump({'pid': os.getpid()}, f)
    return True


def run_method(method, seeds):
    """Trains + evaluates every seed of one method; finished seeds are skipped, unfinished ones resume. A seed that
    another live process has claimed is skipped too (see claim_seed)."""
    rbc = load_rbc()
    rows = []
    for seed in seeds:
        run_dir = run_dir_of(method, seed)
        if (run_dir / 'summary.json').exists():
            print(f"[{method} | seed {seed}] already finished - skipping")
            with open(run_dir / 'summary.json') as f:
                rows.append(summary_row(json.load(f)))
            continue
        if not claim_seed(run_dir):
            print(f"[{method} | seed {seed}] being trained by another process - skipping")
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
