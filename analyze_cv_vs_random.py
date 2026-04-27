import json
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

def analyze_experiments():
    print("Parsing CV vs Random Output Files...")
    results = []
    
    # Files are named eval_results_{Mode}_d{Duration}.json
    files = glob.glob("results/eval_results_*.json")
    if not files:
        print("[ERROR] No evaluation results found. Did you run run_cv_experiments.py?")
        return
        
    for f in files:
        parts = f.replace(".json", "").split("_")
        mode = parts[-2]
        duration = int(parts[-1].replace("d", ""))
        
        try:
            with open(f, "r") as file:
                data = json.load(file)
                
            lane_means = [np.mean(lane) if len(lane) > 0 else 0 for lane in data.get("lane_wait_times", [[],[],[],[]])]
            avg_wait = np.mean(lane_means)
            lane_variance = np.var(lane_means)
            
            throughput = data.get("throughput", [0])[-1] if data.get("throughput") else 0
            
            results.append({
                "Mode": mode,
                "Duration": duration,
                "WaitTime": avg_wait,
                "Throughput": throughput,
                "LaneVariance": lane_variance,
                "Lane0_Wait": lane_means[0],
                "Lane1_Wait": lane_means[1],
                "Lane2_Wait": lane_means[2],
                "Lane3_Wait": lane_means[3]
            })
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    if not results:
        return

    df = pd.DataFrame(results).sort_values(by=["Duration", "Mode"])
    df.to_csv("cv_comparison_results.csv", index=False)
    print("Saved numeric data to cv_comparison_results.csv")
    
    # Filter for standard duration for main bar charts (e.g., 100s, or pick largest)
    target_duration = df["Duration"].mode()[0] if len(df) > 0 else 100
    if 100 in df["Duration"].values:
        target_duration = 100
        
    df_plot = df[df["Duration"] == target_duration]
    
    # 1. Wait Time
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_plot["Mode"], df_plot["WaitTime"], color=["#4CAF50", "#2196F3"])
    plt.title(f"Average Wait Time Comparison ({target_duration}s)")
    plt.ylabel("Wait Time (seconds)")
    plt.bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    plt.savefig("wait_time_comparison.png", dpi=300)
    plt.close()
    
    # 2. Throughput
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_plot["Mode"], df_plot["Throughput"], color=["#FF9800", "#9C27B0"])
    plt.title(f"Throughput Comparison ({target_duration}s)")
    plt.ylabel("Total Vehicles Cleared")
    plt.bar_label(bars, fmt='%.1f')
    plt.tight_layout()
    plt.savefig("throughput_comparison.png", dpi=300)
    plt.close()
    
    # 3. Lane Variance / Fairness
    plt.figure(figsize=(10, 5))
    lanes = ['Lane 0', 'Lane 1', 'Lane 2', 'Lane 3']
    rand_row = df_plot[df_plot["Mode"] == "Random"].iloc[0] if len(df_plot[df_plot["Mode"] == "Random"]) > 0 else None
    cv_row = df_plot[df_plot["Mode"] == "CV"].iloc[0] if len(df_plot[df_plot["Mode"] == "CV"]) > 0 else None
    
    if rand_row is not None and cv_row is not None:
        x = np.arange(len(lanes))
        width = 0.35
        rand_waits = [rand_row["Lane0_Wait"], rand_row["Lane1_Wait"], rand_row["Lane2_Wait"], rand_row["Lane3_Wait"]]
        cv_waits = [cv_row["Lane0_Wait"], cv_row["Lane1_Wait"], cv_row["Lane2_Wait"], cv_row["Lane3_Wait"]]
        
        plt.bar(x - width/2, rand_waits, width, label='Baseline (Random)', color='#9E9E9E')
        plt.bar(x + width/2, cv_waits, width, label='Digital Twin (CV)', color='#E91E63')
        
        plt.title(f"Lane-Wise Imbalance & OS Fairness ({target_duration}s)")
        plt.ylabel("Wait Time per Lane (s)")
        plt.xticks(x, lanes)
        plt.legend()
        plt.tight_layout()
        plt.savefig("lane_variance_plot.png", dpi=300)
        plt.close()
    
    print("Generated:")
    print("- wait_time_comparison.png")
    print("- throughput_comparison.png")
    print("- lane_variance_plot.png")

if __name__ == "__main__":
    analyze_experiments()
