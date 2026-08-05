# train_carbon_agents.py
# Trains and evaluates the Static-Weighted and Dynamic Carbon-Adaptive agents.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

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

def train_agent(reward_type, lambda_val, seed):
    """
    Trains PPO agent for a specific reward formulation and seed.
    """
    name = f"ppo_{reward_type}_seed_{seed}"
    print(f"\n==================================================")
    print(f"Training {reward_type.upper()} Carbon-Aware Agent (Seed {seed})")
    print(f"==================================================")
    
    # Initialize wrapped environment
    env = env_utils.make_env(reward_type=reward_type, lambda_val=lambda_val, seed=seed)
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=config.POLICY_KWARGS,
        seed=seed,
        **config.PPO_PARAMS
    )
    
    # Train
    callback = RLMetricsCallback(verbose=1)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    
    # Save checkpoint
    model_path = os.path.join(config.MODELS_DIR, name)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training log
    train_log_path = os.path.join(config.TABLES_DIR, f"{name}_train_log.csv")
    df_train = pd.DataFrame(callback.training_log)
    df_train.to_csv(train_log_path, index=False)
    
    # Run evaluation trace for final checkpoint
    print(f"Evaluating {reward_type} agent on seed {seed}...")
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
    df_eval.to_csv(os.path.join(config.TABLES_DIR, f"{name}_eval_trace.csv"), index=False)
    
    # Save summary metrics
    summary_path = os.path.join(config.TABLES_DIR, f"{name}_summary.csv")
    summary = pd.DataFrame([{
        'method': f"{reward_type.capitalize()}-weighted PPO",
        'seed': seed,
        'cost': total_cost,
        'carbon': total_carbon,
        'reward': df_train['reward'].iloc[-1] if not df_train.empty else 0.0
    }])
    summary.to_csv(summary_path, index=False)
    print(f"Eval completed: Cost = {total_cost:.2f}, Carbon = {total_carbon:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train carbon-aware PPO agents.")
    parser.add_argument('--seed', type=int, default=None, help="Specific seed to run. If None, runs all configured seeds.")
    parser.add_argument('--lambda_val', type=float, default=1.0, help="Base lambda parameter for static/dynamic rewards.")
    args = parser.parse_args()
    
    seeds = [args.seed] if args.seed is not None else config.SEEDS
    
    for seed in seeds:
        # 1. Train Static Carbon-Aware Agent (Step 3)
        train_agent(reward_type='static', lambda_val=args.lambda_val, seed=seed)
        
        # 2. Train Dynamic Carbon-Adaptive Agent (Step 4)
        train_agent(reward_type='dynamic', lambda_val=args.lambda_val, seed=seed)

if __name__ == "__main__":
    main()
