# run_all_training.py
# Sequential orchestrator to run baselines and carbon-aware agents across all seeds.

import subprocess
import sys

def main():
    print("==================================================")
    print("Orchestrator: Training all baselines and agents...")
    print("==================================================")
    
    # 1. Train Baselines (RBC and Cost-Only PPO across 3 seeds)
    print("\n>>> Part 1: Training RBC and Cost-Only PPO Baselines...")
    sys.stdout.flush()
    try:
        subprocess.run([r".\venv\Scripts\python.exe", "-u", "train_baselines.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during baseline training: {e}")
        sys.exit(1)
        
    # 2. Train Carbon-Aware Agents (Static-weighted and Dynamic PPO across 3 seeds)
    # Using lambda_val = 1.0 (well-tuned carbon baseline from our sweep)
    print("\n>>> Part 2: Training Static and Dynamic Carbon-Aware Agents...")
    sys.stdout.flush()
    try:
        subprocess.run([r".\venv\Scripts\python.exe", "-u", "train_carbon_agents.py", "--lambda_val", "1.0"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during carbon agent training: {e}")
        sys.exit(1)
        
    print("\n==================================================")
    print("Orchestrator: All training runs completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
