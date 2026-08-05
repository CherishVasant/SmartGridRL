# verify_env.py
# Verifies environment setup, loads CityLearn, and checks observation/action spaces.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
import env_utils
import config

def verify():
    print("--- CityLearn Environment Verification ---")
    
    # 1. List available datasets
    try:
        datasets = DataSet.get_names()
        print(f"Available datasets: {datasets}")
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return

    # 2. Check selected dataset from config
    print(f"\nTarget dataset from config: '{config.DATASET_NAME}'")
    if config.DATASET_NAME not in datasets:
        print(f"Error: Selected dataset '{config.DATASET_NAME}' is not available.")
        return
        
    # 3. Create environment using make_env
    print("\nInitializing environment...")
    try:
        env = env_utils.make_env(reward_type='default', seed=42)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    # 4. Check observation and action space shapes
    obs_space = env.observation_space
    act_space = env.action_space
    print(f"\nObservation Space Shape (centralized): {obs_space.shape}")
    print(f"Action Space Shape (centralized): {act_space.shape}")
    print(f"Action range: {act_space.low} to {act_space.high}")

    # 5. Check if carbon intensity is exposed in observations
    base_env = env.env # Access the unwrapped CityLearnEnv
    obs_names = base_env.observation_names[0]
    print(f"\nObservation features size: {len(obs_names)}")
    print(f"Carbon intensity feature name: '{obs_names[config.CARBON_INTENSITY_IDX]}'")
    
    # Verify values at index
    obs, _ = env.reset()
    val = obs[config.CARBON_INTENSITY_IDX]
    print(f"Carbon intensity value sample at step 0: {val:.4f}")
    
    # Let's verify other key indices from config
    print(f"Electricity pricing feature name: '{obs_names[config.PRICING_IDX]}', value sample: {obs[config.PRICING_IDX]:.4f}")
    print(f"Solar generation feature name: '{obs_names[config.SOLAR_GENERATION_IDX]}', value sample: {obs[config.SOLAR_GENERATION_IDX]:.4f}")
    print(f"Electrical storage SoC feature name: '{obs_names[config.SOC_IDX]}', value sample: {obs[config.SOC_IDX]:.4f}")

    print("\nEnvironment setup verified and ready for training baselines!")

if __name__ == "__main__":
    verify()
