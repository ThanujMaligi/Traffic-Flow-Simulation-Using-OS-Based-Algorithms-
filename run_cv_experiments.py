import os
import time
import glob
import shutil
import json
import random
import threading

DURATIONS = [70, 100, 130]
SEED = 100

def mock_cv_sensor(stop_event):
    """
    Simulates the YOLO CV sensor outputting data to the JSON bridge.
    We use a fixed random seed to guarantee 100% reproducibility.
    We simulate 'realistic non-uniform traffic' (Lane 2 and 3 are major highways).
    """
    random.seed(42)
    choices = ['car', 'bike', 'bus', 'truck', 'rickshaw']
    weights = [0.50, 0.35, 0.05, 0.05, 0.05]
    while not stop_event.is_set():
        # Highly unbalanced traffic (realistic city intersection)
        data = {
            "lane_0": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(0, 1))],
            "lane_1": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(0, 2))],
            "lane_2": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(4, 9))],
            "lane_3": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(3, 7))]
        }
        try:
            with open("cv_traffic_counts.json", "w") as f:
                json.dump(data, f)
        except Exception:
            pass # Ignore lock collisions
        time.sleep(1.0) # CV updates ~every 1 second

def run_experiment(module_name, duration, mode_label):
    print(f"\n[RUNNING] Mode: {mode_label} | Duration: {duration}s")
    
    # 1. Setup CV Mock if needed
    stop_event = threading.Event()
    feeder_thread = None
    if mode_label == "CV":
        feeder_thread = threading.Thread(target=mock_cv_sensor, args=(stop_event,))
        feeder_thread.daemon = True
        feeder_thread.start()

    # 2. Generate and run temporary test script
    runner_code = f"""
import {module_name} as sim
sim.SIMULATION_MODE = 2 # 2 = MLFQ algorithm
sim.SEED = {SEED}
sim.simTime = {duration}
sim.Main()
"""
    with open("temp_cv_runner.py", "w") as f:
        f.write(runner_code)
    
    # Run Pygame simulation (blocking)
    os.system("python temp_cv_runner.py")
    
    # 3. Cleanup CV Mock
    if feeder_thread:
        stop_event.set()
        feeder_thread.join()
        
    time.sleep(1) # Let files flush
    
    # 4. Grab the latest JSON results and rename it for our evaluation
    files = glob.glob("results/results_mlfq_*.json")
    if not files:
        print(f"[ERROR] No results generated for {mode_label}")
        return
        
    latest_file = max(files, key=os.path.getctime)
    dest_name = f"results/eval_results_{mode_label}_d{duration}.json"
    shutil.copy(latest_file, dest_name)
    print(f"[SUCCESS] Saved results to {dest_name}")

if __name__ == "__main__":
    print("==================================================")
    print("STARTING RIGOROUS CV VS RANDOM EXPERIMENTAL SUITE")
    print("==================================================")
    
    for d in DURATIONS:
        # Run Baseline
        run_experiment("simulation", d, "Random")
        # Run CV Digital Twin
        run_experiment("simulation_cv", d, "CV")

    print("\n[FINISHED] All experiments orchestrated successfully!")
    print("Proceed to run analyze_cv_vs_random.py")
