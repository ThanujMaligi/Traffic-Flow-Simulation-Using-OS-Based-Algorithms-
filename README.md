# Traffic Flow Simulation Using OS-Based Algorithms

## Overview

This project simulates real-world traffic flow at a four-way intersection using **Operating System scheduling algorithms** combined with **Machine Learning** and **Computer Vision** for intelligent, adaptive signal control.

Traffic lanes are modeled as CPU processes. OS scheduling concepts are applied to manage signal timing, vehicle priority, and resource allocation — mimicking how an OS manages competing processes.

---

## 🧠 Algorithms Implemented

| Algorithm | Role in Simulation |
|---|---|
| **Round Robin (RR)** | Baseline equal time-slot signal scheduling |
| **Multi-Level Feedback Queue (MLFQ)** | Adaptive signal timing based on queue length |
| **Banker's Algorithm** | Deadlock avoidance at intersections |
| **Priority Scheduling** | Emergency vehicle preemption |
| **Random Forest** | ML-based traffic flow prediction |
| **XGBoost** | Congestion level classification |
| **YOLOv8 (CV)** | Real-time vehicle detection and counting |

---

## 📁 Project Structure

```text
TFS OS Project/
│
├── simulation.py                  # Core simulation with OS scheduling algorithms
├── simulation_cv.py               # CV-enhanced simulation using YOLOv8 vehicle detection
├── old_simulation.py              # Baseline simulation (no OS scheduling)
│
├── AILayer/                       # Machine Learning models and inference
├── models/                        # Trained ML model files
│
├── cv_sensor.py                   # Computer vision traffic sensor module
├── cv_traffic_counts.json         # CV-detected vehicle count reference data
├── yolov8n.pt                     # YOLOv8 nano model weights
├── traffic_video.mp4              # Sample traffic footage for CV testing
│
├── Banglore_traffic_Dataset.csv   # Real-world Bangalore traffic dataset
├── dataset_integration_runner_ml.py  # ML dataset pipeline runner
│
├── run_experiments.py             # Main experiment runner
├── run_all_models.py              # Runs all ML models
├── run_ml_experiments.py          # ML-specific experiment runner
├── run_cv_experiments.py          # CV pipeline experiment runner
├── run_duration_test.py           # Simulation duration stress tests
├── run_validation_pcu.py          # PCU-based validation runner
│
├── analyze_results.py             # Result analysis and comparison
├── analyze_cv_vs_random.py        # CV vs random count analysis
├── plot_results.py                # Visualization and plotting
│
├── results/                       # Per-seed experiment result JSONs
├── results.json                   # Aggregated experiment results
├── statistical_summary.json       # Statistical summary across all runs
├── cv_comparison_results.csv      # CV vs random comparison data
│
├── Images/                        # Vehicle and signal sprite assets
├── Outputs/                       # Simulation screenshots and outputs
│
├── OS report.pdf                  # Full project report
├── Final Project Presentation.pptx  # Presentation slides
└── README.md                      # Project documentation
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ThanujMaligi/Traffic-Flow-Simulation-Using-OS-Based-Algorithms-.git
cd Traffic-Flow-Simulation-Using-OS-Based-Algorithms-

# Install dependencies
pip install pygame numpy pandas scikit-learn xgboost ultralytics opencv-python matplotlib
```

---

## 🚀 Running the Simulation

```bash
# Run the main OS-based simulation
python simulation.py

# Run the Computer Vision enhanced simulation
python simulation_cv.py

# Run all experiments across algorithms
python run_experiments.py

# Run ML model experiments
python run_ml_experiments.py

# Run CV experiments
python run_cv_experiments.py

# Analyze results
python analyze_results.py

# Plot results
python plot_results.py
```

---

## 📊 Results & Visualizations

The project compares multiple scheduling strategies across metrics like:
- **Throughput** (vehicles/second)
- **Average waiting time** per lane
- **Emergency vehicle response time**
- **Lane load distribution**
- **Banker's algorithm deadlock prevention**

Key output plots generated:
- `throughput_plot.png` — Algorithm throughput comparison
- `waiting_time_plot.png` — Waiting time distributions
- `lane_load_plot.png` — Per-lane load analysis
- `banker_decisions.png` — Banker's algorithm resource decisions
- `emergency_response_plot.png` — Emergency vehicle preemption timing
- `algorithm_usage_plot.png` — Algorithm usage breakdown

---

## 🎓 Course Details

- **Course**: Operating Systems — `23AID213`
- **Guide**: Pooja Gowda
- **Team**:
  - [Thanuj](https://github.com/ThanujMaligi)
  - [Nikhilesh](https://github.com/mikey9029)
  - [Jayavardhan](https://github.com/JAYYYYYYYYYYYYYYYYYYYYYYYYYY)

---

## 🙏 Acknowledgements

Thanks to **Pooja Gowda ma'am** for her continuous guidance and support throughout the project.
Inspired by OS scheduling concepts applied to real-world traffic management systems.
