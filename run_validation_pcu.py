import os
import sys
import time
import json
import random
import threading
import glob
import pandas as pd

print("==================================================")
print("PCU ENGINE VALIDATION SUITE")
print("==================================================")

SEED = 100
DURATION = 100

def mock_cv_sensor(stop_event):
    """
    Feeds traffic into simulation_cv.py.
    Simulates heavy traffic with heterogeneous composition (buses, trucks, bikes, cars).
    Occasionally spawns an ambulance to test IRQ and PIP logic.
    """
    random.seed(SEED)
    choices = ['car', 'bike', 'bus', 'truck', 'rickshaw']
    weights = [0.40, 0.30, 0.15, 0.10, 0.05] # Mix with significant heavy vehicles
    
    tick = 0
    while not stop_event.is_set():
        # Unbalanced heavy traffic
        data = {
            "lane_0": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(0, 1))],
            "lane_1": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(0, 2))],
            "lane_2": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(4, 9))], # High congestion lane
            "lane_3": [random.choices(choices, weights=weights)[0] for _ in range(random.randint(3, 7))]
        }
        
        # Inject Ambulance at specific ticks (e.g., tick 20 and 50)
        if tick == 20 or tick == 50:
            lane_to_inject = random.choice([0, 1, 2, 3])
            data[f"lane_{lane_to_inject}"].append("ambulance")
            print(f"[VALIDATION FEEDER] Injecting Emergency Ambulance in Lane {lane_to_inject + 1}")
            
        try:
            with open("cv_traffic_counts.json", "w") as f:
                json.dump(data, f)
        except Exception:
            pass
            
        tick += 1
        time.sleep(1.0)

# Start CV sensor feeder
stop_event = threading.Event()
feeder_thread = threading.Thread(target=mock_cv_sensor, args=(stop_event,))
feeder_thread.daemon = True
feeder_thread.start()

print(f"[BOOT] Starting simulation_cv.py in MLFQ mode for {DURATION}s...")

runner_code = f"""
import simulation_cv as sim
sim.SIMULATION_MODE = 2 # MLFQ
sim.SEED = {SEED}
sim.simTime = {DURATION}
sim.Main()
"""

with open("temp_cv_runner.py", "w") as f:
    f.write(runner_code)

# Run the simulation
exit_code = os.system("python temp_cv_runner.py")

# Stop the feeder
stop_event.set()
feeder_thread.join()

print("\n==================================================")
print("ANALYZING RESULTS")
print("==================================================")

if exit_code != 0:
    print(f"[FAIL] Simulation crashed with exit code {exit_code}")
    sys.exit(1)

# Check results/pcu_metrics_*.csv
pcu_csv_files = glob.glob("results/pcu_metrics_*.csv")
if not pcu_csv_files:
    print("[FAIL] No PCU metrics CSV exported!")
    sys.exit(1)

latest_csv = max(pcu_csv_files, key=os.path.getctime)
print(f"[SUCCESS] PCU metrics file found: {latest_csv}")

df = pd.read_csv(latest_csv)
print("\nPCU Weights exported:")
for col in df.columns:
    if "PCU_Weight_" in col:
        print(f"  {col}: {df[col].iloc[0]}")

print("\nPCU Load and Congestion Stats:")
for lane in range(1, 5):
    max_load = df[f"L{lane}_ActivePCULoad"].max()
    avg_load = df[f"L{lane}_ActivePCULoad"].mean()
    max_cong = df[f"L{lane}_CongestionScore"].max()
    avg_cong = df[f"L{lane}_CongestionScore"].mean()
    print(f"  Lane {lane} -> Max PCU Load: {max_load:.2f} | Avg PCU Load: {avg_load:.2f} | Max Congestion: {max_cong:.2f} | Avg Congestion: {avg_cong:.2f}")

print("\n==================================================")
print("VALIDATION SUCCESSFUL!")
print("==================================================")
