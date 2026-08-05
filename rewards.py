# rewards.py
# Implements the baseline and dynamic reward functions for the smart grid environment.

import numpy as np
import config

def get_step_metrics(env):
    """
    Extracts the operational cost and carbon emissions of the step that was just completed.
    In CityLearn 2.1.2, building cost/emission arrays accumulate history, so the current
    step's values are the last elements in these arrays.
    """
    # Sum across all buildings for the current timestep (index -1)
    cost = sum(float(b.net_electricity_consumption_cost[-1]) for b in env.buildings)
    carbon = sum(float(b.net_electricity_consumption_emission[-1]) for b in env.buildings)
    return cost, carbon

def get_carbon_intensity(env):
    """
    Extracts the marginal carbon intensity for the step that was just completed.
    Since env.time_step is incremented after env.step(), the step that just completed
    corresponds to env.time_step - 1.
    """
    # Read carbon intensity of the step that just finished
    step_idx = max(0, env.time_step - 1)
    return float(env.buildings[0].carbon_intensity._carbon_intensity[step_idx])

# 1. Cost-only Reward: R = -cost
def cost_only_reward(env, action):
    cost, _ = get_step_metrics(env)
    return -cost

# 2. Static-weighted Carbon-aware Reward: R = -cost - lambda * carbon
def static_carbon_reward(env, action, lambda_val):
    cost, carbon = get_step_metrics(env)
    return -cost - (lambda_val * carbon)

# 3. Dynamic Carbon-adaptive Reward: R = -cost - lambda(intensity) * carbon
def dynamic_carbon_reward(env, action, lambda_base, carbon_min, carbon_max):
    cost, carbon = get_step_metrics(env)
    intensity = get_carbon_intensity(env)
    
    # Calculate normalized intensity in [0, 1]
    if carbon_max > carbon_min:
        norm_intensity = (intensity - carbon_min) / (carbon_max - carbon_min)
    else:
        norm_intensity = 0.0
        
    # Clip to [0, 1] to guard against out-of-bounds testing data
    norm_intensity = np.clip(norm_intensity, 0.0, 1.0)
    
    # Scaling function phi(intensity) = norm_intensity (linear scaling)
    # This means lambda dynamically ranges from 0 (at min carbon) to lambda_base (at max carbon)
    lambda_dynamic = lambda_base * norm_intensity
    
    return -cost - (lambda_dynamic * carbon)
