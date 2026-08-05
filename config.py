# config.py
# Single source of truth for paths, hyperparameters, and environment settings.

import os

# Dataset & Environment Setup
DATASET_NAME = 'citylearn_challenge_2022_phase_1'
NUM_BUILDINGS = 5
SEEDS = [42, 100, 2026]

# Indices for observation variables (based on building 0 observation list)
CARBON_INTENSITY_IDX = 19
NON_SHIFTABLE_LOAD_IDX = 20
SOLAR_GENERATION_IDX = 21
SOC_IDX = 22
NET_ELECTRICITY_CONSUMPTION_IDX = 23
PRICING_IDX = 24

# PPO Training Hyperparameters
# Using standard tuned hyperparameters for SB3 PPO in energy dispatch tasks.
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

# Neural Network Architecture
POLICY_KWARGS = {
    'net_arch': {
        'pi': [64, 64],
        'vf': [64, 64]
    }
}

# Training steps (about 5.7 episodes since 1 episode is 8760 steps)
TOTAL_TIMESTEPS = 50_000

# Directory Setup
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Create directories if they do not exist
for d in [PLOTS_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Configuration loaded successfully. Results directories prepared.")
