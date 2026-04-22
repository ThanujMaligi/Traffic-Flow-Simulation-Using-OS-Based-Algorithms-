import os
import sys
import time
import json
import numpy as np
import glob

# Durations to test
DURATIONS = [70, 100, 130]
MODES = {0: "RR", 1: "MVF", 2: "MLFQ"}
SEED = 100

def run_sim(mode, duration, seed):
    # We modify simulation.py on the fly or just use a wrapper that sets simTime
    # Since we can't easily pass it as arg without more edits, we'll use a small trick:
    # We'll write a temp runner script for each run.
    runner_code = f"""
import simulation as sim
sim.SIMULATION_MODE = {mode}
sim.SEED = {seed}
sim.simTime = {duration}
sim.Main()
"""
    with open("temp_runner.py", "w") as f:
        f.write(runner_code)
    
    print(f"[RUNNING] Mode: {MODES[mode]}, Duration: {duration}s")
    os.system(f"python temp_runner.py {mode} {seed}")
    time.sleep(1)

def get_latest_result(mode_name, duration):
    files = glob.glob(f"results/results_{mode_name.lower()}_t{duration}_seed{SEED}_*.json")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        data = json.load(f)
    
    lane_means = [np.mean(lane) for lane in data["lane_wait_times"] if lane]
    avg_wait = np.mean(lane_means) if lane_means else 0
    throughput = data["throughput"][-1] if data["throughput"] else 0
    return {"wait": avg_wait, "tp": throughput}

# Main Execution
print("Starting Duration Comparison Experiment...")
for d in DURATIONS:
    for m in MODES:
        run_sim(m, d, SEED)

# AI-RF is the winner, let's test it too
for d in DURATIONS:
    # AI-RF usually runs via dataset_integration_runner_ml.py
    runner_code = f"""
import simulation as sim
import dataset_integration_runner_ml as runner
sim.SIMULATION_MODE = 2
sim.SEED = {SEED}
sim.simTime = {d}
runner.MODEL_TYPE = "rf"
sim.Main()
"""
    with open("temp_runner.py", "w") as f:
        f.write(runner_code)
    
    print(f"[RUNNING] Mode: AI-RF, Duration: {d}s")
    os.system(f"python temp_runner.py 2 {SEED} rf")
    time.sleep(1)

print("\n--- EXPERIMENT COMPLETE ---")
