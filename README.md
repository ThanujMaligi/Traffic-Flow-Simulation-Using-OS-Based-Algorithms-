# Traffic Flow Simulation Using OS-Based Algorithms

## Overview
This project simulates traffic flow at a four-way intersection, comparing system performance before and after implementing Operating System (OS) algorithms like First-Come-First-Serve (FCFS) and priority scheduling. It prioritizes emergency vehicles (e.g., ambulances) to reduce delays, using Python and Pygame for visualization. The simulation models traffic lanes as CPU units, applying concepts like multithreading and dynamic resource allocation.

## Course Details
- **Course Name**: Operating Systems  
- **Course Code**: 23AID213  
- **Team Members**:
  - Given Name Surname (GitHub: [github.com/Thanuj](https://github.com/ThanujMaligi))
  - Given Name Surname (GitHub: [github.com/Nikhilesh](https://github.com/mikey9029)
  - Given Name Surname (GitHub: [github.com/Jayavardhan](https://github.com/JAYYYYYYYYYYYYYYYYYYYYYYYYYY))
- **Project Guide**: Pooja Gowda

---

## Prerequisites
Before running the simulation, ensure you have the following installed:
- Python 3.8 or higher
- Pygame library
- A compatible operating system (Windows, macOS, or Linux)

---


The Pygame window will open, displaying the four-way intersection simulation. You can observe traffic flow:

Before OS Algorithms: Round-robin scheduling with no priority for emergency vehicles.

After OS Algorithms: FCFS and priority scheduling prioritize ambulances, with maximum vehicles first for regular traffic.



traffic-flow-simulation/
│
├── simulation.py            # Main simulation code with OS-based scheduling
├── old_simulation.py        # Older simulation code (without OS algorithms)
├── images/                  # Screenshots, output images, graphs
├── ppt/                     # Final presentation slides
└── report/                  # Project report, documentation, references





## Installation

Follow these steps to set up the project on your local machine:

```bash
pip install pygame
```

```bash
# Clone the Repository
git clone https://github.com/your-repo/traffic-flow-simulation.git
cd traffic-flow-simulation
```



# Set Up a Virtual Environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```



Running the Simulation
To run the traffic flow simulation:
```
# Navigate to the project directory
cd traffic-flow-simulation

# Run the main simulation script
python main.py


```

Acknowledgments
Inspired by OS scheduling concepts from our coursework.

Thanks to Pooja Gowda for her continuous guidance and support.
