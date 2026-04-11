import os
import time

# Experiment Configuration (Mirrors run_experiments.py)
MODES = [0, 1, 2] # 0: RR, 1: MVF, 2: MLFQ
MODE_NAMES = ["ML-RR", "ML-MVF", "ML-MLFQ"]
RUNS_PER_MODE = 10 

print("="*60)
print("[INFO] STARTING ML-ENHANCED RESEARCH BATCH (10 RUNS PER MODE)")
print(f"Total ML Runs: {len(MODES) * RUNS_PER_MODE}")
print("="*60)

for mode in MODES:
    print(f"\n[TESTING] Testing ML-Enhanced Strategy: {MODE_NAMES[mode]}")
    for run in range(RUNS_PER_MODE):
        # Use the SAME seed mapping as the baseline runner for fair comparison
        seed = run + 100 
        print(f"   [ML RUN {run+1}/{RUNS_PER_MODE}] starting with SEED {seed}...")
        
        # Call the ML Wrapper instead of the raw simulation
        os.system(f"python dataset_integration_runner_ml.py {mode} {seed}")
        
        time.sleep(1)

print("\n" + "="*60)
print("[SUCCESS] ML EXPERIMENTS COMPLETE")
print("Now run 'python analyze_results.py' to compare Baseline vs ML.")
print("="*60)
