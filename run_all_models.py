import os
import time

MODELS = ["lr", "dt", "rf", "xgb"]
RUNS_PER_MODEL = 10 # Total 40 runs

print("="*60)
print("[INFO] STARTING MULTI-MODEL COMPETITION")
print(f"Models: {MODELS}")
print(f"Total Runs: {len(MODELS) * RUNS_PER_MODEL}")
print("="*60)

# We'll use Mode 2 (MLFQ) as the core baseline for all models
SIM_MODE = 2 

for model in MODELS:
    print(f"\n[TESTING] Testing Model: {model.upper()}")
    for run in range(RUNS_PER_MODEL):
        seed = run + 100
        # Args: mode, seed, model_type
        command = f"python dataset_integration_runner_ml.py {SIM_MODE} {seed} {model}"
        os.system(command)
        time.sleep(1)

print("\n" + "="*60)
print("[SUCCESS] ALL MODEL EXPERIMENTS COMPLETE")
print("Run 'python analyze_results.py' to see the winner!")
print("="*60)
