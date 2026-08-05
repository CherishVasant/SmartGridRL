# env_utils.py
# Shared utilities to construct CityLearn environments with custom rewards.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import gymnasium as gym
import numpy as np
from citylearn.citylearn import CityLearnEnv
import config
import rewards

class CityLearnStableBaselinesWrapper(gym.Env):
    """
    Gymnasium wrapper that adapts CityLearnEnv to work with Stable-Baselines3.
    It:
    - Converts CityLearn's old Gym API (reset returns obs; step returns 4 values)
      to Gymnasium API (reset returns obs, info; step returns 5 values).
    - Flattens multi-agent observation/action lists to centralized spaces.
    - Explicitly converts old gym.spaces.Box to gymnasium.spaces.Box to resolve SB3 type assertions.
    - Applies custom reward functions (cost-only, static, or dynamic).
    - Tracks cumulative episode-level metrics (operational cost, carbon emissions, and reward)
      for evaluation and logging.
    """
    def __init__(self, env, reward_type='default', lambda_val=None):
        super().__init__()
        self.env = env
        self.reward_type = reward_type
        self.lambda_val = lambda_val
        
        # Centralized spaces: extract old Gym Box spaces and convert to Gymnasium Box spaces
        old_obs_space = env.observation_space[0]
        old_act_space = env.action_space[0]
        
        self.observation_space = gym.spaces.Box(
            low=old_obs_space.low,
            high=old_obs_space.high,
            dtype=old_obs_space.dtype
        )
        
        self.action_space = gym.spaces.Box(
            low=old_act_space.low,
            high=old_act_space.high,
            dtype=old_act_space.dtype
        )
        
        # Precompute carbon min/max from dataset for dynamic reward normalization
        carbon_intensities = self.env.buildings[0].carbon_intensity._carbon_intensity
        self.carbon_min = float(np.min(carbon_intensities))
        self.carbon_max = float(np.max(carbon_intensities))
        
        self.episode_cost = 0.0
        self.episode_carbon = 0.0
        self.episode_reward = 0.0
        print(f"Wrapper initialized: reward_type={reward_type}, lambda={lambda_val}")
        print(f"Carbon intensity bounds detected: min={self.carbon_min:.4f}, max={self.carbon_max:.4f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.random_seed = seed
        
        # Reset tracking metrics
        self.episode_cost = 0.0
        self.episode_carbon = 0.0
        self.episode_reward = 0.0
        
        obs = self.env.reset()
        return np.array(obs[0], dtype=np.float32), {}

    def step(self, action):
        # Action is a numpy array of shape (5,)
        # CityLearn expects action inside a list: [action]
        obs, default_reward, done, info = self.env.step([action])
        
        # Track cost and carbon step-by-step
        cost, carbon = rewards.get_step_metrics(self.env)
        self.episode_cost += cost
        self.episode_carbon += carbon
        
        # Calculate custom reward
        if self.reward_type == 'cost_only':
            reward = rewards.cost_only_reward(self.env, action)
        elif self.reward_type == 'static':
            assert self.lambda_val is not None, "lambda_val must be specified for static reward"
            reward = rewards.static_carbon_reward(self.env, action, self.lambda_val)
        elif self.reward_type == 'dynamic':
            assert self.lambda_val is not None, "lambda_val (base) must be specified for dynamic reward"
            reward = rewards.dynamic_carbon_reward(
                self.env, action, self.lambda_val, self.carbon_min, self.carbon_max
            )
        else:
            # Default CityLearn reward: sum of building rewards
            reward = float(sum(default_reward))
            
        self.episode_reward += reward
        terminated = done
        truncated = False
        
        # Inject metrics into info dict for callbacks or evaluation
        info['step_metrics'] = {
            'cost': cost,
            'carbon': carbon,
            'reward': reward
        }
        if done:
            info['episode_metrics'] = {
                'cost': self.episode_cost,
                'carbon': self.episode_carbon,
                'reward': self.episode_reward
            }
        
        return np.array(obs[0], dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_env(reward_type='default', lambda_val=None, seed=None):
    """
    Helper function to initialize and wrap CityLearn environment.
    """
    env = CityLearnEnv(schema=config.DATASET_NAME, central_agent=True)
    wrapped_env = CityLearnStableBaselinesWrapper(env, reward_type=reward_type, lambda_val=lambda_val)
    
    if seed is not None:
        wrapped_env.reset(seed=seed)
        
    return wrapped_env
