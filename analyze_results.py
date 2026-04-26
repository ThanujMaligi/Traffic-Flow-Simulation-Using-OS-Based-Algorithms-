import json
import numpy as np
import glob
import os
import math
import re
from scipy.stats import ttest_rel

def extract_metrics(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        lane_means = [np.mean(lane) for lane in data["lane_wait_times"] if lane]
        avg_wait = np.mean(lane_means) if lane_means else 0
        throughput = data["throughput"][-1] if data["throughput"] else 0
        emergency = np.mean(data["emergency_response"]) if data["emergency_response"] else 0
        return avg_wait, throughput, emergency
    except Exception:
        return None

def analyze_all():
    print("="*85)
    print("--- MULTI-MODEL LEADERBOARD: TRAFFIC AI COMPETITION ---")
    print("="*85)
    print(f"{'Rank':<5} | {'Model':<10} | {'Wait (s)':<10} | {'Throughput':<12} | {'Stability':<10}")
    print("-" * 85)

    summary_data = {}
    
    # Modes to check
    baseline_modes = ["RR", "MVF", "MLFQ"]
    ai_models = ["LR", "DT", "RF", "XGB"]

    all_results = []

    # 1. Process Baseline
    for mode in baseline_modes:
        files = glob.glob(f"results/results_{mode.lower()}_*.json")
        if not files: continue
        
        waits = []
        tps = []
        for f in files:
            m = extract_metrics(f)
            if m:
                waits.append(m[0])
                tps.append(m[1])
        
        if waits:
            mean_w = np.mean(waits)
            std_w = np.std(waits, ddof=1) if len(waits) > 1 else 0
            mean_tp = np.mean(tps)
            
            all_results.append({
                "name": mode,
                "wait": mean_w,
                "tp": mean_tp,
                "std": std_w,
                "is_ai": False
            })

    # 2. Process AI Models
    # Note: These are usually tagged as results_{model}_mlfq_*.json
    for model in ai_models:
        files = glob.glob(f"results/results_{model.lower()}_mlfq_*.json")
        if not files: continue
        
        waits = []
        tps = []
        for f in files:
            m = extract_metrics(f)
            if m:
                waits.append(m[0])
                tps.append(m[1])
        
        if waits:
            mean_w = np.mean(waits)
            std_w = np.std(waits, ddof=1) if len(waits) > 1 else 0
            mean_tp = np.mean(tps)
            
            all_results.append({
                "name": f"AI-{model}",
                "wait": mean_w,
                "tp": mean_tp,
                "std": std_w,
                "is_ai": True
            })

    # Sort by Wait Time (Ascending - Lower is better)
    all_results.sort(key=lambda x: x["wait"])

    for i, res in enumerate(all_results):
        stability = (res["std"] / res["wait"]) if res["wait"] > 0 else 0
        rank = i + 1
        indicator = "*" if res["is_ai"] else " "
        print(f"{rank:<5} | {res['name'] + indicator:<10} | {res['wait']:<10.2f} | {res['tp']:<12.1f} | {stability:<10.3f}")

    # Identify Best Model
    if all_results:
        best = all_results[0]
        print("\n" + "="*85)
        print(f"WINNER: {best['name']} (Wait Time: {best['wait']:.2f}s)")
        print("="*85)

    # Save summary for plotting
    plot_summary = {res["name"]: res for res in all_results}
    with open("statistical_summary.json", "w") as f:
        json.dump(plot_summary, f)

if __name__ == "__main__":
    analyze_all()
