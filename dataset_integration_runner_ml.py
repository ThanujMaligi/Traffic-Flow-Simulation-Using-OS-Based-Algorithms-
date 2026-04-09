import os
import sys
import datetime
import json
import time
import threading

# Import the target module
import simulation as sim

# Import ML inference
from AILayer.inference import predict_green_time

# --- SAFE THRASHING CONTROL CONFIGURATION ---
last_switch_time = 0
MIN_HOLD_TIME = 6   # seconds before switching allowed
LANE_AFFINITY_THRESHOLD = 15 # Prefer current lane if queue > 15

# Configuration
MODEL_TYPE = "rf" # Default
if len(sys.argv) > 3:
    MODEL_TYPE = sys.argv[3]

print(f"[INFO] Setting up SAFE THRASHING CONTROL (Model: {MODEL_TYPE.upper()})...")

# Store original functions for wrapping
original_select_next_signal = sim.select_next_signal
original_setTime = sim.setTime

# --- 1. Patched select_next_signal (Safe Switching Control ONLY) ---
def safe_thrashing_aware_select_next_signal():
    global last_switch_time
    
    # Get candidate from the original scheduler
    task = original_select_next_signal()
    
    # Only evaluate switching for Normal/Restore tasks
    if task['type'] not in ['NORMAL', 'RESTORE']:
        return task
        
    target_signal = task['target']
    
    # Detect Switch Attempt
    if target_signal != sim.currentGreen:
        current_time = time.time()
        time_since_switch = current_time - last_switch_time
        
        # A. Minimum Hold Time Enforcement
        if time_since_switch < MIN_HOLD_TIME:
            if sim.timeElapsed % 5 == 0:
                print(f"[THRASHING BLOCK] Switch to {target_signal+1} prevented. Elapsed: {time_since_switch:.1f}s")
            # Force keep currentGreen by returning a HOLD task
            return {'type': 'HOLD', 'source': '[SAFE-THRASHING] Min Hold Active'}
            
        # B. Safe Lane Affinity (Optional addition)
        current_queue = sim.signals[sim.currentGreen].active_count
        if current_queue > LANE_AFFINITY_THRESHOLD:
            if sim.timeElapsed % 5 == 0:
                print(f"[LANE AFFINITY] Keeping {sim.currentGreen+1} due to high queue ({current_queue})")
            return {'type': 'HOLD', 'source': '[SAFE-THRASHING] Affinity Active'}

        # Switch Allowed
        print(f"[SWITCH ALLOWED] Transitioning to Signal {target_signal+1}")
        last_switch_time = time.time() # Update ONLY on switch
        
    return task

# --- 2. Patched setTime (ML Only - NO PENALTY) ---
def ml_setTime_pure():
    """
    Standard ML prediction without any timing modifications or penalties.
    """
    lane = sim.currentGreen
    features = sim.get_dataset_features(lane)
    
    # Predict using selected model type
    predicted_time = predict_green_time(features, MODEL_TYPE)
    effective_time = int(predicted_time)

    if lane < len(sim.signals):
        sim.signals[lane].green = effective_time
        sim.log_event("AI_SCHEDULER", f"Lane {lane+1} | {MODEL_TYPE.upper()} Prediction: {effective_time}s")
    
    return effective_time

# --- 3. FORCE INJECT INTO SIMULATION GLOBALS ---
sim.setTime = ml_setTime_pure
sim.select_next_signal = safe_thrashing_aware_select_next_signal
sim.MODEL_NAME = MODEL_TYPE

# Ensure internal module dict is updated
sim.__dict__['setTime'] = ml_setTime_pure
sim.__dict__['select_next_signal'] = safe_thrashing_aware_select_next_signal
sim.__dict__['MODEL_NAME'] = MODEL_TYPE

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_integration_runner_ml.py <mode> [seed] [model_type]")
        sys.exit(1)
        
    # Initialize clock
    last_switch_time = time.time()
    sim.Main()
