# evaluate.py
# Loads trained policies, runs final evaluations, computes self-consumption,
# and generates summary tables, Pareto frontier, and training curve plots.

# Import torch FIRST to prevent DLL conflict (WinError 1114) with pandas on Windows
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC

import config
import env_utils

# Use seaborn aesthetics for clean, premium plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def run_evaluation_episode(method, seed):
    """
    Runs a deterministic evaluation episode and returns detailed metrics.
    """
    print(f"Evaluating {method} on seed {seed}...")
    
    # 1. Initialize environment based on method
    if method == 'RBC':
        env = CityLearnEnv(schema=config.DATASET_NAME, central_agent=False)
        env.random_seed = seed
        obs = env.reset()
        agent = BasicRBC(env)
    else:
        # PPO Agents (Cost-Only, Static, Dynamic)
        reward_type = 'cost_only'
        if method == 'Static':
            reward_type = 'static'
        elif method == 'Dynamic':
            reward_type = 'dynamic'
            
        # Standard env wrapper
        env = env_utils.make_env(reward_type=reward_type, lambda_val=1.0, seed=seed)
        
        # Load PPO checkpoint
        model_name = f"ppo_{reward_type}_seed_{seed}"
        model_path = os.path.join(config.MODELS_DIR, model_name)
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        model = PPO.load(model_path, env=env)
        obs, _ = env.reset(seed=seed)

    # 2. Episode loop
    total_cost = 0.0
    total_carbon = 0.0
    
    # Track building consumption and solar profiles for self-consumption rate
    # CityLearn stores arrays of building histories.
    # At the end of the episode, we can read them directly from building objects.
    time_steps = env.time_steps if method == 'RBC' else env.env.time_steps
    
    for step in range(time_steps):
        if method == 'RBC':
            actions = agent.predict(obs)
            obs, reward, done, info = env.step(actions)
        else:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
        if done:
            break

    # 3. Retrieve building instances to compute emissions and self-consumption
    buildings = env.buildings if method == 'RBC' else env.env.buildings
    
    # Net cost and carbon emissions are the sums of histories (excluding index 0 initial placeholder)
    total_cost = sum(float(sum(b.net_electricity_consumption_cost[1:])) for b in buildings)
    total_carbon = sum(float(sum(b.net_electricity_consumption_emission[1:])) for b in buildings)
    
    # Self-consumption calculation:
    # Exported electricity per building at step t: max(0, -net_electricity_consumption)
    # Self-consumed solar: solar_generation - exported
    # Rate = total_self_consumed / total_solar_generated
    total_solar = 0.0
    total_exported = 0.0
    
    for b in buildings:
        net_cons = np.array(b.net_electricity_consumption[1:], dtype=np.float32)
        solar = np.array(b.solar_generation[1:], dtype=np.float32)
        
        exported = np.maximum(0.0, -net_cons)
        
        total_solar += float(np.sum(solar))
        total_exported += float(np.sum(exported))
        
    self_consumption_rate = (total_solar - total_exported) / total_solar if total_solar > 0 else 1.0

    return {
        'method': method,
        'seed': seed,
        'cost': total_cost,
        'carbon': total_carbon,
        'self_consumption': self_consumption_rate
    }

def generate_plots(results_df):
    """
    Generates Pareto Frontier scatter plot and training reward curves.
    """
    print("\nGenerating evaluation plots...")
    
    # Compute averages across seeds
    summary_avg = results_df.groupby('method').agg({
        'cost': ['mean', 'std'],
        'carbon': ['mean', 'std'],
        'self_consumption': 'mean'
    }).reset_index()
    summary_avg.columns = ['method', 'cost_mean', 'cost_std', 'carbon_mean', 'carbon_std', 'self_consumption_mean']
    
    # ----------------------------
    # Plot 1: Pareto Frontier
    # ----------------------------
    plt.figure(figsize=(10, 7))
    colors = {
        'RBC': '#7f8c8d',           # gray
        'Cost-Only': '#e74c3c',     # red
        'Static': '#f39c12',        # orange
        'Dynamic': '#2ecc71'        # green
    }
    
    # Plot individual seeds as small markers
    for _, row in results_df.iterrows():
        plt.scatter(
            row['carbon'], row['cost'],
            color=colors.get(row['method'], '#3498db'),
            alpha=0.4, s=60, edgecolors='none'
        )
        
    # Plot average points with error bars
    for _, row in summary_avg.iterrows():
        plt.errorbar(
            row['carbon_mean'], row['cost_mean'],
            xerr=row['carbon_std'], yerr=row['cost_std'],
            fmt='o', markersize=12, label=row['method'],
            color=colors.get(row['method'], '#3498db'),
            capsize=6, elinewidth=2, markeredgecolor='black', markeredgewidth=1
        )
        
    plt.xlabel('Carbon Emissions (kg CO2-eq)')
    plt.ylabel('Operational Cost ($)')
    plt.title('Cost-Carbon Pareto Frontier (Smart Grid Dispatch)')
    plt.legend(title="Method", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'pareto_frontier.png'), dpi=300)
    plt.close()
    print("- Saved pareto_frontier.png")

    # ----------------------------
    # Plot 2: Training Curves
    # ----------------------------
    plt.figure(figsize=(11, 6))
    
    # Loop over PPO methods
    for method_type in ['cost_only', 'static', 'dynamic']:
        method_label = {
            'cost_only': 'Cost-Only',
            'static': 'Static',
            'dynamic': 'Dynamic'
        }[method_type]
        
        method_color = colors[method_label]
        
        # Load all seed training logs for this method
        seed_logs = []
        for seed in config.SEEDS:
            log_path = os.path.join(config.TABLES_DIR, f"ppo_{method_type}_seed_{seed}_train_log.csv")
            if os.path.exists(log_path):
                df_seed = pd.read_csv(log_path)
                seed_logs.append(df_seed)
                
        if not seed_logs:
            continue
            
        # Align rows by episode (assuming all training runs completed same number of episodes)
        min_len = min(len(df) for df in seed_logs)
        aligned_rewards = np.zeros((min_len, len(config.SEEDS)))
        steps = seed_logs[0]['step'].iloc[:min_len].values
        
        for s_idx, df in enumerate(seed_logs):
            aligned_rewards[:, s_idx] = df['reward'].iloc[:min_len].values
            
        mean_rewards = np.mean(aligned_rewards, axis=1)
        std_rewards = np.std(aligned_rewards, axis=1)
        
        plt.plot(steps, mean_rewards, label=method_label, color=method_color, linewidth=2.5)
        plt.fill_between(
            steps, mean_rewards - std_rewards, mean_rewards + std_rewards,
            color=method_color, alpha=0.15
        )
        
    plt.xlabel('Training Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('PPO Multi-Seed Training Curves (Rewards)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'training_curves.png'), dpi=300)
    plt.close()
    print("- Saved training_curves.png")

def main():
    print("--- Running Final Evaluation for All Methods and Seeds ---")
    
    methods = ['RBC', 'Cost-Only', 'Static', 'Dynamic']
    all_results = []
    
    for method in methods:
        for seed in config.SEEDS:
            try:
                res = run_evaluation_episode(method, seed)
                all_results.append(res)
            except Exception as e:
                print(f"Error evaluating {method} on seed {seed}: {e}")
                
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(config.TABLES_DIR, 'evaluation_raw_results.csv'), index=False)
    
    # Compute base RBC metrics for comparison
    rbc_costs = df_results[df_results['method'] == 'RBC']['cost'].values
    rbc_carbon = df_results[df_results['method'] == 'RBC']['carbon'].values
    avg_rbc_cost = np.mean(rbc_costs)
    avg_rbc_carbon = np.mean(rbc_carbon)
    
    # Compile Summary Table
    summary_list = []
    for method in methods:
        method_df = df_results[df_results['method'] == method]
        if method_df.empty:
            continue
            
        cost_mean = method_df['cost'].mean()
        cost_std = method_df['cost'].std()
        
        carbon_mean = method_df['carbon'].mean()
        carbon_std = method_df['carbon'].std()
        
        self_cons_mean = method_df['self_consumption'].mean() * 100  # Percentage
        
        # Calculate carbon emissions avoided compared to RBC average
        carbon_avoided = avg_rbc_carbon - carbon_mean
        
        summary_list.append({
            'Method': method,
            'Cost ($)': f"{cost_mean:.2f} ± {cost_std:.2f}",
            'Emissions (kg CO2)': f"{carbon_mean:.2f} ± {carbon_std:.2f}",
            'Emissions Avoided (kg)': f"{carbon_avoided:.2f}",
            'Self-Consumption Rate (%)': f"{self_cons_mean:.2f}%"
        })
        
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(config.TABLES_DIR, 'summary_table.csv'), index=False)
    
    # Save markdown summary
    md_table = summary_df.to_markdown(index=False)
    with open(os.path.join(config.TABLES_DIR, 'summary_table.md'), 'w') as f:
        f.write("# Summary of Results (Averages over 3 Seeds)\n\n")
        f.write(md_table)
        
    print("\n--- Summary of Results (Averages over 3 Seeds) ---")
    print(md_table)
    
    # Generate Plots
    generate_plots(df_results)
    print("\nEvaluation successfully completed. Results written to results folder.")

if __name__ == "__main__":
    main()
