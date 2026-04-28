import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_leaderboard():
    summary_path = "statistical_summary.json"
    if not os.path.exists(summary_path):
        print(f"[ERROR] {summary_path} not found.")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Sort data by wait time for consistent plotting
    items = sorted(data.values(), key=lambda x: x["wait"])
    names = [item["name"] for item in items]
    waits = [item["wait"] for item in items]
    tps = [item["tp"] for item in items]
    is_ai = [item["is_ai"] for item in items]

    colors = ['#2ecc71' if ai else '#3498db' for ai in is_ai]

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Final Multi-Model Competition: Identifying the Best Traffic AI", fontsize=16, fontweight='bold')

    # 1. Wait Time
    bars1 = ax1.bar(names, waits, color=colors)
    ax1.set_ylabel('Avg Waiting Time (s)')
    ax1.set_title('Fairness Ranking (Lower is Better)')
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

    # 2. Throughput
    bars2 = ax2.bar(names, tps, color=colors)
    ax2.set_ylabel('Total Vehicles Cleared')
    ax2.set_title('Efficiency Ranking (Higher is Better)')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                    Line2D([0], [0], color='#3498db', lw=4)]
    ax1.legend(custom_lines, ['AI Enhanced', 'OS Baseline'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "Outputs/multi_model_competition.png"
    if not os.path.exists("Outputs"): os.makedirs("Outputs")
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Competition plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_leaderboard()
