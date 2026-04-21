import os
import time
import sys

# Experiment Configuration
MODES = [0, 1, 2]  # 0: Round Robin, 1: MVF (Density), 2: MLFQ (Advanced OS)
MODE_NAMES = ["Round-Robin", "MVF (Density)", "MLFQ (OS-Kernel)"]
RUNS_PER_MODE = 10 # Enforced 10 runs for statistical validation

print("="*50)
print("🚦 STARTING TRAFFIC SIMULATION RESEARCH EXPERIMENTS")
print(f"Total Runs: {len(MODES) * RUNS_PER_MODE}")
print("="*50)

for mode in MODES:
    print(f"\n🚀 Testing Strategy: {MODE_NAMES[mode]}")
    for run in range(RUNS_PER_MODE):
        seed = run + 100 # Consistent seed per iteration across all modes
        print(f"   [RUN {run+1}/{RUNS_PER_MODE}] starting with SEED {seed}...")
        # Run simulation with mode and seed as CLI arguments
        # We use os.system for a clean process isolation per run
        os.system(f"python simulation.py {mode} {seed}")
        
        # Brief cooldown between runs to ensure files are flushed and OS resources cleared
        time.sleep(1)

print("\n" + "="*50)
print("✅ EXPERIMENTS COMPLETE")
print("Run 'python analyze_results.py' to process statistical data.")
print("="*50)
