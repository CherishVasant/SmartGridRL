# train_static_sweep.py
# Runs a hyperparameter sweep over lambda values for the static carbon reward on Seed 42.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import os
import pandas as pd
from stable_baselines3 import PPO
import config
import env_utils

def evaluate_agent(model, env):
    """
    Evaluates the agent for 1 full episode and returns cumulative cost and carbon.
    """
    obs, _ = env.reset(seed=42)
    total_cost = 0.0
    total_carbon = 0.0
    
    for step in range(env.env.time_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        metrics = info['step_metrics']
        total_cost += metrics['cost']
        total_carbon += metrics['carbon']
        
        if terminated:
            break
            
    return total_cost, total_carbon

def main():
    print("--- Starting Static Lambda Sweep on Seed 42 ---")
    
    lambdas = [0.2, 1.0, 5.0]
    sweep_results = []
    
    for l_val in lambdas:
        print(f"\nTraining static carbon-aware agent with lambda = {l_val}...")
        env = env_utils.make_env(reward_type='static', lambda_val=l_val, seed=42)
        
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=config.POLICY_KWARGS,
            seed=42,
            **config.PPO_PARAMS
        )
        
        # Train for 100,000 steps
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
        
        # Evaluate
        eval_cost, eval_carbon = evaluate_agent(model, env)
        print(f"Lambda = {l_val} Evaluation: Cost = {eval_cost:.2f}, Carbon = {eval_carbon:.2f}")
        
        sweep_results.append({
            'lambda': l_val,
            'cost': eval_cost,
            'carbon': eval_carbon
        })
        
        # Save model checkpoint
        model.save(os.path.join(config.MODELS_DIR, f'ppo_static_lambda_{l_val}_seed_42'))
        
    df = pd.DataFrame(sweep_results)
    df.to_csv(os.path.join(config.TABLES_DIR, 'static_lambda_sweep_results.csv'), index=False)
    
    print("\n--- Sweep Results ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
