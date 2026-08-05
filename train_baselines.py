# train_baselines.py
# Trains and evaluates Rule-Based Controller (RBC) and Cost-Only PPO baselines.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC

import config
import env_utils

class RLMetricsCallback(BaseCallback):
    """
    Callback to record training progress (rewards, costs, carbon) on episode completion.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.training_log = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]
                    if 'episode_metrics' in info:
                        metrics = info['episode_metrics']
                        self.episode_count += 1
                        self.training_log.append({
                            'episode': self.episode_count,
                            'step': self.num_timesteps,
                            'reward': metrics['reward'],
                            'cost': metrics['cost'],
                            'carbon': metrics['carbon']
                        })
                        if self.verbose > 0:
                            print(f"Episode {self.episode_count} (Step {self.num_timesteps}): "
                                  f"Reward={metrics['reward']:.2f}, Cost={metrics['cost']:.2f}, Carbon={metrics['carbon']:.2f}")
        return True

def run_rbc(seed):
    """
    Runs the deterministic Rule-Based Controller (RBC) for 1 full episode.
    """
    print(f"\n--- Running RBC Baseline (Seed {seed}) ---")
    env = CityLearnEnv(schema=config.DATASET_NAME, central_agent=False)
    env.random_seed = seed
    obs = env.reset()
    agent = BasicRBC(env)
    
    total_cost = 0.0
    total_carbon = 0.0
    step_data = []
    
    for step in range(env.time_steps):
        actions = agent.predict(obs)
        obs, reward, done, info = env.step(actions)
        
        cost = sum(float(b.net_electricity_consumption_cost[-1]) for b in env.buildings)
        carbon = sum(float(b.net_electricity_consumption_emission[-1]) for b in env.buildings)
        
        total_cost += cost
        total_carbon += carbon
        
        # Log key values for analysis
        intensity = obs[0][config.CARBON_INTENSITY_IDX]
        pricing = obs[0][config.PRICING_IDX]
        solar = sum(o[config.SOLAR_GENERATION_IDX] for o in obs)
        load = sum(o[config.NON_SHIFTABLE_LOAD_IDX] for o in obs)
        soc = sum(o[config.SOC_IDX] for o in obs) / len(obs)
        
        step_data.append({
            'step': step,
            'cost': cost,
            'carbon': carbon,
            'carbon_intensity': intensity,
            'pricing': pricing,
            'solar_gen': solar,
            'non_shiftable_load': load,
            'avg_soc': soc,
            'action_b0': actions[0][0]
        })
        
        if done:
            break
            
    df = pd.DataFrame(step_data)
    df.to_csv(os.path.join(config.TABLES_DIR, f'rbc_seed_{seed}_eval_trace.csv'), index=False)
    
    # Save high-level metrics summary
    summary = pd.DataFrame([{
        'method': 'RBC',
        'seed': seed,
        'cost': total_cost,
        'carbon': total_carbon,
        'reward': 0.0  # Default reward placeholder
    }])
    summary.to_csv(os.path.join(config.TABLES_DIR, f'rbc_seed_{seed}_summary.csv'), index=False)
    print(f"RBC Seed {seed} Summary: Cost = {total_cost:.2f}, Carbon = {total_carbon:.2f}")

def train_ppo_cost_only(seed):
    """
    Trains PPO agent optimizing for operational cost only (reward = -cost).
    """
    print(f"\n--- Training PPO Cost-Only Agent (Seed {seed}) ---")
    env = env_utils.make_env(reward_type='cost_only', seed=seed)
    
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=config.POLICY_KWARGS,
        seed=seed,
        **config.PPO_PARAMS
    )
    
    callback = RLMetricsCallback(verbose=1)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    
    # Save model and training log
    model.save(os.path.join(config.MODELS_DIR, f'ppo_cost_only_seed_{seed}'))
    
    df_train = pd.DataFrame(callback.training_log)
    df_train.to_csv(os.path.join(config.TABLES_DIR, f'ppo_cost_only_seed_{seed}_train_log.csv'), index=False)
    
    # Run evaluation trace for final checkpoint
    print(f"Evaluating Cost-Only agent (Seed {seed})...")
    obs, _ = env.reset(seed=seed)
    total_cost = 0.0
    total_carbon = 0.0
    step_data = []
    
    for step in range(env.env.time_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        metrics = info['step_metrics']
        total_cost += metrics['cost']
        total_carbon += metrics['carbon']
        
        step_data.append({
            'step': step,
            'cost': metrics['cost'],
            'carbon': metrics['carbon'],
            'carbon_intensity': obs[config.CARBON_INTENSITY_IDX],
            'pricing': obs[config.PRICING_IDX],
            'action_b0': action[0]
        })
        
        if terminated:
            break
            
    df_eval = pd.DataFrame(step_data)
    df_eval.to_csv(os.path.join(config.TABLES_DIR, f'ppo_cost_only_seed_{seed}_eval_trace.csv'), index=False)
    
    # Save summary metrics
    summary = pd.DataFrame([{
        'method': 'Cost-Only PPO',
        'seed': seed,
        'cost': total_cost,
        'carbon': total_carbon,
        'reward': df_train['reward'].iloc[-1] if not df_train.empty else 0.0
    }])
    summary.to_csv(os.path.join(config.TABLES_DIR, f'ppo_cost_only_seed_{seed}_summary.csv'), index=False)
    print(f"Cost-Only PPO Seed {seed} Final Eval: Cost = {total_cost:.2f}, Carbon = {total_carbon:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train baselines for Smart Grid Dispatch.")
    parser.add_argument('--seed', type=int, default=None, help="Specific seed to run. If None, runs all configured seeds.")
    args = parser.parse_args()
    
    seeds = [args.seed] if args.seed is not None else config.SEEDS
    
    for seed in seeds:
        # Run RBC Baseline
        run_rbc(seed)
        
        # Train PPO Cost-Only agent
        train_ppo_cost_only(seed)

if __name__ == "__main__":
    main()
