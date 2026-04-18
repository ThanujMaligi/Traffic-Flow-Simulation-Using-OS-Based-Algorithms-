import random
import math
import time
import threading
import pygame
import sys
import os
import datetime
import csv
import pandas as pd
from collections import deque
import json
import numpy as np

# --- SCIENTIFIC DETERMINISM (IEEE REQUIREMENT) ---
SEED = int(time.time()) # Default
if len(sys.argv) > 2:
    try:
        SEED = int(sys.argv[2])
    except ValueError:
        pass

random.seed(SEED)
np.random.seed(SEED)
print(f"[BOOT] Random Seed initialized to: {SEED}")

# Load dataset ONCE at top
try:
    df = pd.read_csv("Banglore_traffic_Dataset.csv")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    df = None

# Lane Mapping
lane_map = {
    0: "100 Feet Road",
    1: "CMH Road",
    2: "Marathahalli Bridge",
    3: "Sony World Junction"
}

# Global Dataset Index for Stable Sampling
dataset_index = 0

def get_dataset_features(lane):
    global dataset_index
    
    # Safe Defaults
    default_features = {
        "volume": 10000,
        "speed": 40,
        "congestion": 50,
        "incidents": 0,
        "compliance": 80,
        "pedestrians": 100,
        "weather": "clear",
        "capacity": 50,
        "tti": 1.0,
        "area": "unknown"
    }

    if df is None:
        return default_features
    
    road = lane_map.get(lane)
    try:
        if road:
            subset = df[df['Road/Intersection Name'] == road]
            if not subset.empty:
                row = subset.iloc[dataset_index % len(subset)]
            else:
                row = df.iloc[dataset_index % len(df)]
        else:
            row = df.iloc[dataset_index % len(df)]
        
        # Increment index for time-consistent sampling
        dataset_index += 1

        return {
            "volume": row["Traffic Volume"],
            "speed": row["Average Speed"],
            "congestion": row["Congestion Level"],
            "incidents": row["Incident Reports"],
            "compliance": row["Traffic Signal Compliance"],
            "pedestrians": row["Pedestrian and Cyclist Count"],
            "weather": str(row["Weather Conditions"]).lower(),
            "capacity": row["Road Capacity Utilization"],
            "tti": row["Travel Time Index"],
            "area": row["Area Name"]
        }
    except Exception as e:
        log_event("ERROR", f"Dataset sampling failed: {e}")
        return default_features


# Lane Features Cache to prevent redundant dataset sampling
lane_features_cache = {}

def get_speed_factor(lane):
    if lane not in lane_features_cache:
        lane_features_cache[lane] = get_dataset_features(lane)
    
    features = lane_features_cache[lane]
    base_speed_val = features["speed"]
    
    # Tiered Speed Scaling (Visual Fix for Research Comparison)
    if base_speed_val < 25:
        factor = 0.7
    elif base_speed_val < 35:
        factor = 1.0
    elif base_speed_val < 45:
        factor = 1.5
    else:
        factor = 2.0
        
    return max(0.8, min(2.5, factor))

# Logging System
def log_event(type, msg):
    if type == "FIXED":
        return
    print(f"[{time.strftime('%H:%M:%S')}] [{type}] {msg}")

# Simulation Modes for Comparison
MODE_ROUND_ROBIN = 0
MODE_MVF = 1
MODE_MLFQ = 2
SIMULATION_MODE = MODE_MLFQ # Default
MODEL_NAME = "" # For AI Model identification in filenames

# CLI Support for Multi-Mode Experiments
if len(sys.argv) > 1:
    try:
        SIMULATION_MODE = int(sys.argv[1])
        print(f"[BOOT] Simulation Mode set to: {SIMULATION_MODE}")
        print(f"[MODE ACTIVE] {['ROUND_ROBIN', 'MVF', 'MLFQ'][SIMULATION_MODE]}")
    except ValueError:
        print("[ERROR] Invalid mode passed. Defaulting to MLFQ.")

current_algorithm = "Idle"

# Phase 4: Synchronization & Metrics
# Phase 4: Synchronization & Metrics
data_lock = threading.Lock()
vehicle_lock = threading.Lock()
is_emergency_active = False
metrics = {
    'total_wait_time': 0,
    'total_vehicles_cleared': 0,
    'emergency_response_times': [],
    'safety_violations_prevented': 0,
    'start_time': time.time()
}

# Banker's Algorithm Global State
INTERSECTION_CAPACITY_PCU = 50.0
banker_available = INTERSECTION_CAPACITY_PCU
banker_allocation = [0.0, 0.0, 0.0, 0.0]
banker_need = [0.0, 0.0, 0.0, 0.0]
banker_max = [INTERSECTION_CAPACITY_PCU] * 4
# Performance Tracking Globals
time_data = []
throughput_data = []
noOfSignals = 4
lane_wait_times = [[] for _ in range(noOfSignals)]
lane_load_data = [[] for _ in range(noOfSignals)]

# Validation Check for Initialization
if len(lane_load_data) == noOfSignals:
    log_event("FIXED", f"System Initialized | Signals: {noOfSignals}")
else:
    log_event("WARNING", "Metric array mismatch during init")
algo_usage = {
    "MLFQ (Aging)": 0,
    "MLFQ (Shockwave Promotion)": 0,
    "MLFQ (Oldest Process First)": 0,
    "MLFQ (MVF)": 0,
    "MLFQ (Round Robin)": 0,
    "Priority Inheritance Protocol": 0,
    "Interrupt Handling (Emergency IRQ)": 0,
    "Banker's Override": 0,
    "Context Switching": 0
}
banker_stats = {
    "safe_count": 0,
    "unsafe_count": 0,
    "decision_log": []
}

class VehiclePCB:
    def __init__(self, vehicle_id, vehicle_type, direction, arrival_time):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.arrival_time = arrival_time
        self.wait_time = 0
        self.priority = 0
        self.state = "waiting" # waiting, moving, cleared
        
        # Queue state telemetry fields
        self.accumulated_wait_time = 0.0
        self.queue_start_time = None
        self.is_stopped = False
        self.is_slow = False
        self.is_queued = False

    def get_current_wait_time(self):
        """Calculates current wait time in seconds (OS wait-register representation)"""
        t = self.accumulated_wait_time
        if self.queue_start_time is not None:
            t += time.time() - self.queue_start_time
        return t


# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 5
defaultMaximum = 20

signals = []
simTime = 100
timeElapsed = 0
waiting_for_cv = True
last_telemetry_time = 0

currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Phase 3: Preemption & Context Switching Globals
is_in_yellow_phase = False # Atomic Yellow Phase Guard
saved_context = None
last_interrupt_time = 0
min_execution_time = 5  # Seconds a lane must run before it can be preempted
interrupt_cooldown = 8  # Increased to reduce emergency spam
ambulance_queue = deque()
max_emergency_duration = 15 # Seconds
last_movement_time = time.time()
WATCHDOG_TIMEOUT = 12 # Reset if no movement for 12 seconds
last_scheduler = "Idle"
opf_lock_time = 0

# --- ADVANCED EMERGENCY IRQ KERNEL GLOBALS ---
# Analogous to CPU Interrupt Service Routines (ISRs) and Hardware Interrupts (IRQs)
emergency_interrupt = False
active_emergency_lane = -1
emergency_start_time = 0
last_ambulance_spawn_time = 0
emergency_cooldown = 25 # Seconds between emergency events (prevent IRQ storm)
total_emergency_events = 0
preemption_durations = []
interrupted_signal_stats = []
pip_logged_directions = set()

# --- CONGESTION-AWARE QUEUE INTELLIGENCE LAYER GLOBALS ---
# Analogous to queue structures, thread state tables, and wait-time registers in OS kernels
queue_length = [0, 0, 0, 0]
stopped_vehicles_count = [0, 0, 0, 0]
slow_vehicles_count = [0, 0, 0, 0]
average_wait_time = [0.0, 0.0, 0.0, 0.0]
max_wait_time = [0.0, 0.0, 0.0, 0.0]
queue_growth_rate = [0.0, 0.0, 0.0, 0.0]
previous_queue_length = [0, 0, 0, 0]
congestion_scores = [0.0, 0.0, 0.0, 0.0]

# Tunable mathematical coefficients for Congestion Score Engine
ALPHA_QUEUE = 0.4
BETA_WAIT = 0.3
GAMMA_GROWTH = 0.3

# Thread-safe telemetry log list to dump CSV data on exit
queue_metrics_history = []

# --- Passenger Car Unit (PCU) Configuration ---
vehicle_pcu_weights = {
    'bike': 0.75,
    'car': 1.0,
    'rickshaw': 1.5,
    'bus': 3.0,
    'truck': 3.0,
    'ambulance': 2.5
}
pcu_load = [0.0, 0.0, 0.0, 0.0]
pcu_queued_load = [0.0, 0.0, 0.0, 0.0]
pcu_metrics_history = []


# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5
ambulanceTime = 1.5  # Faster crossing time for ambulances

# Count of vehicles at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfAmbulances = 0
noOfLanes = 2

# Red signal time at which vehicles are detected
detectionTime = 5

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5, 'ambulance': 3.0}  # Ambulance has higher speed

# Coordinates of start
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

#In-Memory Storage with vehicles Dictionary:
vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike', 5: 'ambulance'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# --- Slope Analysis (Legacy - replaced by pandas integration) ---
# growth_factor = compute_growth_factor()
growth_factor = 1.0 # Default fallback

# --- Banker's Algorithm: Resource Request Logic ---
def update_banker_state():
    global banker_available, banker_allocation, banker_need
    with vehicle_lock:
        alloc = [0.0, 0.0, 0.0, 0.0]
        need = [0.0, 0.0, 0.0, 0.0]
        for i in range(noOfSignals):
            direction = directionNumbers[i]
            waiting = 0.0
            in_intersection = 0.0
            for lane in range(3):
                for v in vehicles[direction][lane]:
                    v_pcu = vehicle_pcu_weights.get(v.vehicleClass, 1.0)
                    if v.crossed == 0:
                        waiting += v_pcu
                    else:
                        # Check if still in intersection (boundary check)
                        if direction == 'right' and v.x < 1400: in_intersection += v_pcu
                        elif direction == 'down' and v.y < 800: in_intersection += v_pcu
                        elif direction == 'left' and v.x > 0: in_intersection += v_pcu
                        elif direction == 'up' and v.y > 0: in_intersection += v_pcu
            alloc[i] = in_intersection
            need[i] = waiting
        
        with data_lock:
            banker_allocation = alloc
            banker_need = need
            banker_available = max(0.0, INTERSECTION_CAPACITY_PCU - sum(alloc))

def is_safe_state():
    work = banker_available
    finish = [False] * noOfSignals
    safe_sequence = []
    
    for _ in range(noOfSignals):
        found = False
        for i in range(noOfSignals):
            if not finish[i] and banker_need[i] <= work:
                work += banker_allocation[i]
                finish[i] = True
                safe_sequence.append(i + 1)
                found = True
                break
        if not found:
            break
            
    if all(finish):
        log_event("BANKER", f"Safe sequence found: {safe_sequence}")
        return True
    return False

def banker_resource_request(signal_idx):
    global banker_available, banker_allocation, banker_need
    
    # Request is the number of vehicles waiting in that lane
    request = banker_need[signal_idx]
    
    if request == 0:
        return True # Safe to grant green if no one is there
        
    with data_lock:
        if banker_allocation[signal_idx] > 0.01:
            return True # Avoid double allocation
            
        # Check 1: Request <= Need (Always true by definition here)
        # Check 2: Request <= Available
        if request > banker_available:
            log_event("BANKER", f"Request DENIED for Signal {signal_idx+1}: Request ({request:.2f}) > Available ({banker_available:.2f})")
            return False
            
        # Temporary Allocation
        banker_available = max(0.0, banker_available - request)
        banker_allocation[signal_idx] += request
        banker_need[signal_idx] -= request
        
        if is_safe_state():
            log_event("BANKER", f"Request GRANTED for Signal {signal_idx+1}")
            banker_stats["safe_count"] += 1
            banker_stats["decision_log"].append({"time": timeElapsed, "lane": signal_idx+1, "decision": "SAFE"})
            return True
        else:
            # Rollback
            banker_available += request
            banker_allocation[signal_idx] -= request
            banker_need[signal_idx] += request
            log_event("BANKER", f"Request DENIED for Signal {signal_idx+1}: Unsafe state detected. Rollback.")
            banker_stats["unsafe_count"] += 1
            banker_stats["decision_log"].append({"time": timeElapsed, "lane": signal_idx+1, "decision": "UNSAFE"})
            return False

# --- Priority Inheritance Protocol ---
# Prevents Priority Inversion where lower-priority processes block a high-priority interrupt
def apply_priority_inheritance():
    global pip_logged_directions
    with vehicle_lock:
        for direction in directionNumbers.values():
            for lane in range(3):
                queue = vehicles[direction][lane]
                # Find the index of the first ambulance that hasn't crossed yet
                amb_idx = -1
                for idx, v in enumerate(queue):
                    if hasattr(v, 'pcb') and v.pcb.vehicle_type == 'ambulance' and v.crossed == 0:
                        amb_idx = idx
                        break
                
                # Inherit priority to everyone in front of the ambulance in this lane
                if amb_idx >= 0:
                    # Log PIP activation once per event per direction
                    if direction not in pip_logged_directions:
                        log_event("PIP", f"Priority inherited by leading vehicles in {direction} corridor (OS Priority Inheritance)")
                        pip_logged_directions.add(direction)
                        
                    # Boost speed for all vehicles in front of and including the ambulance
                    for idx in range(amb_idx + 1):
                        v = queue[idx]
                        if v.crossed == 0 and hasattr(v, 'pcb'):
                            v.pcb.priority = 10 # Critical priority promotion
                            v.speed = speeds.get(v.vehicleClass, 2.0) * 1.5 # Boost speed to 1.5x to clear path
                else:
                    # Reset priority if no ambulance is active in this lane
                    for v in queue:
                        if hasattr(v, 'pcb') and v.crossed == 0:
                            v.pcb.priority = 0
                            v.speed = speeds.get(v.vehicleClass, 2.0)
# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

stops = {'right': [580, 580, 580], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425}, 'up': {'x': 695, 'y': 400}}
rotationAngle = 3

# Gap between vehicles
gap = 8 # Reduced for higher density flow
gap2 = 15

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        self.waitTime = 0  # Added for Aging Mechanism
        self.priority = 0  # 0 for Normal, 1 for Aged
        # Explicit Accounting System
        self.spawned_count = 0
        self.exited_count = 0
        self.active_count = 0

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        
        # Realism Fix: Use dataset-driven speed factor
        base_speed = speeds[vehicleClass]
        speed_factor = get_speed_factor(direction_number)
        self.speed = base_speed * speed_factor
        
        # Stability Fix: Clamp speed safely
        self.speed = max(1.0, min(self.speed, 10.0))
        
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)
        
        # Initialize PCB for Phase 4
        self.index = len(vehicles[direction][lane]) 
        vehicle_id = f"{direction}_{self.index}"
        self.pcb = VehiclePCB(vehicle_id, vehicleClass, direction, time.time())
        self.moved_distance = self.speed


        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop.get(direction, 580)
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop.get(direction, 810)
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop.get(direction, 320)
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop.get(direction, 545)
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        
        # Final Registration (Atomic to prevent race conditions)
        with vehicle_lock:
            vehicles[direction][lane].append(self)
        
        with data_lock:
            if direction_number < len(signals):
                signals[direction_number].spawned_count += 1
                signals[direction_number].active_count += 1
                log_event("FIXED", f"Vehicle registration safe for Index {direction_number}")
            
            if vehicleClass == 'ambulance':
                global emergency_interrupt, active_emergency_lane, emergency_start_time, total_emergency_events
                emergency_interrupt = True
                active_emergency_lane = direction_number
                emergency_start_time = time.time()
                total_emergency_events += 1
                
                # Save the active signal state before preemption context switch
                interrupted_signal_stats.append({
                    'interrupted_signal': currentGreen + 1,
                    'remaining_green': signals[currentGreen].green,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                ambulance_queue.append((direction_number, self))
                # Hardware Interrupt Analogy (IRQ)
                log_event("IRQ", f"Emergency Vehicle Detected in Lane {direction_number + 1} (Hardware Interrupt)")
                log_event("IRQ", "Requesting Immediate Signal Preemption (CPU Preemption)")

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

#Vehicle Movement Synchronization in Vehicle.move():
    def move(self):
        global last_movement_time
        # Keep speed stable (already set in init or signal change)
        pass

        self.moved_distance = 0.0

        # 🚑 Emergency Override Movement: Ensure ambulances move when green
        if self.vehicleClass == 'ambulance' and self.crossed == 0:
            if currentGreen == self.direction_number:
                self.speed = speeds['ambulance'] * 1.5 # Boost
            else:
                return # Stop if not green

        old_x, old_y = self.x, self.y

        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                with data_lock:
                    vehicles[self.direction]['crossed'] += 1
                    # Safe Metric Update
                    if self.direction_number < len(signals):
                        signals[self.direction_number].exited_count += 1
                        signals[self.direction_number].active_count -= 1
                        log_event("FIXED", f"Atomic accounting safe for Index {self.direction_number}")
                    
                    if self.vehicleClass == 'ambulance':
                        # Remove from queue if it was the head
                        if ambulance_queue and ambulance_queue[0][1] == self:
                            ambulance_queue.popleft()
                            log_event("EMERGENCY", f"Ambulance {self.pcb.vehicle_id} cleared intersection")
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    # Dynamic Gap Safety
                    current_gap = gap2 if self.crossed == 0 else 6
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - current_gap) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        # Dynamic Gap Safety
                        current_gap = gap2
                        if self.crossed == 1:
                            current_gap = 6 # Tighter inside intersection
                        else:
                            current_gap = 12 # Safer outside
                            
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - current_gap) or \
                           self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2):
                            self.y += self.speed
            else:
                if (self.crossed == 1 or self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0)) and \
                   (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                with data_lock:
                    vehicles[self.direction]['crossed'] += 1
                    # Safe Metric Update
                    if self.direction_number < len(signals):
                        signals[self.direction_number].exited_count += 1
                        signals[self.direction_number].active_count -= 1
                        log_event("FIXED", f"Atomic accounting safe for Index {self.direction_number}")
                    
                    if self.vehicleClass == 'ambulance':
                        if ambulance_queue and ambulance_queue[0][1] == self:
                            ambulance_queue.popleft()
                            log_event("EMERGENCY", f"Ambulance {self.pcb.vehicle_id} cleared intersection")
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                           self.y < (vehicles[self.direction][self.lane][self.index - 1].y - gap2):
                            self.x -= self.speed
            else:
                if (self.crossed == 1 or self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0)) and \
                   (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                with data_lock:
                    vehicles[self.direction]['crossed'] += 1
                    # Safe Metric Update
                    if self.direction_number < len(signals):
                        signals[self.direction_number].exited_count += 1
                        signals[self.direction_number].active_count -= 1
                        log_event("FIXED", f"Atomic accounting safe for Index {self.direction_number}")
                    
                    if self.vehicleClass == 'ambulance':
                        if ambulance_queue and ambulance_queue[0][1] == self:
                            ambulance_queue.popleft()
                            log_event("EMERGENCY", f"Ambulance {self.pcb.vehicle_id} cleared intersection")
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                           self.x > (vehicles[self.direction][self.lane][self.index - 1].x + gap2):
                            self.y -= self.speed
            else:
                if (self.crossed == 1 or self.x >= self.stop or (currentGreen == 2 and currentYellow == 0)) and \
                   (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                with data_lock:
                    vehicles[self.direction]['crossed'] += 1
                    # Safe Metric Update
                    if self.direction_number < len(signals):
                        signals[self.direction_number].exited_count += 1
                        signals[self.direction_number].active_count -= 1
                        log_event("FIXED", f"Atomic accounting safe for Index {self.direction_number}")
                    
                    if self.vehicleClass == 'ambulance':
                        if ambulance_queue and ambulance_queue[0][1] == self:
                            ambulance_queue.popleft()
                            log_event("EMERGENCY", f"Ambulance {self.pcb.vehicle_id} cleared intersection")
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index - 1].x - vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width - gap2) or \
                           self.y > (vehicles[self.direction][self.lane][self.index - 1].y + gap2):
                            self.x += self.speed
            else:
                if (self.crossed == 1 or self.y >= self.stop or (currentGreen == 3 and currentYellow == 0)) and \
                   (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y -= self.speed
        
        if self.x != old_x or self.y != old_y:
            with data_lock:
                last_movement_time = time.time()

        # Compute exact moved distance in this tick (frame)
        self.moved_distance = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)

        # Update PCB State
        if self.crossed == 1:
            if self.pcb.state == "waiting":
                self.pcb.wait_time = time.time() - self.pcb.arrival_time
                with data_lock:
                    if self.direction_number < len(lane_wait_times):
                        lane_wait_times[self.direction_number].append(self.pcb.wait_time)
                        log_event("FIXED", f"Safe wait time record for Index {self.direction_number}")
                    metrics['total_wait_time'] += self.pcb.wait_time
                    metrics['total_vehicles_cleared'] += 1
            self.pcb.state = "cleared"
            # Clean up queue tracking on crossing
            if self.pcb.queue_start_time is not None:
                self.pcb.accumulated_wait_time += (time.time() - self.pcb.queue_start_time)
                self.pcb.queue_start_time = None
        else:
            self.pcb.state = "waiting"


def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfAmbulances, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime, ambulanceTime
    noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfAmbulances = 0, 0, 0, 0, 0, 0
    # Use explicit counts for vehicle classification
    for i in range(3):
        for j in range(len(vehicles[directionNumbers[currentGreen]][i])):
            vehicle = vehicles[directionNumbers[currentGreen]][i][j]
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'bike': noOfBikes += 1
                elif vclass == 'car': noOfCars += 1
                elif vclass == 'bus': noOfBuses += 1
                elif vclass == 'truck': noOfTrucks += 1
                elif vclass == 'rickshaw': noOfRickshaws += 1
                elif vclass == 'ambulance': noOfAmbulances += 1
    
    # --- Dynamic Time Quantum (Density-Based) ---
    total_v = signals[currentGreen].active_count
    density_val = total_v / (noOfLanes * 20.0) 
    density_factor = int(density_val * 15)
    base_time = 5
    
    # --- Dataset Feature-Based Adjustment ---
    features = get_dataset_features(currentGreen)
    # Refresh cache for the current lane on signal change
    lane_features_cache[currentGreen] = features
    adjustment = 0

    # Congestion
    if features["congestion"] > 70:
        adjustment += 5 # Reduced from +6
    elif features["congestion"] > 40:
        adjustment += 4
    else:
        adjustment -= 2

    # Speed
    if features["speed"] < 30:
        adjustment += 3 # Reduced from +4
    elif features["speed"] > 50:
        adjustment -= 2

    # Pedestrians
    if features["pedestrians"] > 150:
        adjustment += 3

    # Incidents
    if features["incidents"] > 0:
        adjustment += 3 # Reduced from +4

    # Road Capacity Utilization
    if features["capacity"] > 90:
        adjustment += 4 # Reduced from +6
    elif features["capacity"] > 70:
        adjustment += 2 # Reduced from +3

    # Travel Time Index
    if features["tti"] > 1.5:
        adjustment += 3 # Reduced from +5
    elif features["tti"] > 1.2:
        adjustment += 2

    # Area Name
    area = str(features["area"]).lower()
    if "indiranagar" in area:
        adjustment += 1 # Reduced from +2
    elif "whitefield" in area:
        adjustment += 2 # Reduced from +3
    elif "koramangala" in area:
        adjustment += 1 # Reduced from +2

    # Weather
    if "rain" in features["weather"] or "storm" in features["weather"]:
        adjustment += 4
    elif "fog" in features["weather"]:
        adjustment += 2

    # Compliance
    if features["compliance"] < 70:
        adjustment += 2

    # Soft Normalization (Avoid saturation in short simulation)
    adjustment = min(adjustment, 10)
    log_event("DEBUG", f"Adjustment Applied: {adjustment}")

    # Final Timing Calculation
    greenTime = base_time + density_factor + adjustment
    
    # Heavy Lane Boost (Fix for overloaded lanes - scaled for 70s run)
    if currentGreen < len(signals) and signals[currentGreen].active_count > 15:
        greenTime += 3
        log_event("DEBUG", "Heavy Lane Boost applied (+3s)")

    # Lane Load Feedback
    lane_load = sum(len(vehicles[directionNumbers[currentGreen]][i]) for i in range(3))
    if lane_load > 15:
        greenTime += 3
    elif lane_load > 8:
        greenTime += 1

    # Clamp Final Output (5s to 15s) - REQUIRED FOR 70s TOTAL TIME
    greenTime = max(5, min(15, int(greenTime)))
        
    if currentGreen < len(signals):
        signals[currentGreen].green = greenTime
        log_event("FIXED", f"Safe timing update for Index {currentGreen}")
    
    # Log Dataset Features and Resulting Timing
    log_event("DATASET", f"Lane {currentGreen+1} | Features: {features} | GreenTime: {greenTime}")
    
    return greenTime

def checkAmbulances():
    valid = []
    with data_lock:
        while ambulance_queue:
            idx, v = ambulance_queue[0]
            # Remove stale ambulances (already crossed)
            if v.crossed == 1:
                ambulance_queue.popleft()
                continue
            # Only consider if actually waiting near intersection
            if v.crossed == 0:
                valid.append((idx, v))
            break # Only head of queue matters for IRQ
    return valid

def get_pip_lane():
    with vehicle_lock:
        for i in range(noOfSignals):
            direction = directionNumbers[i]
            for lane in range(3):
                for v in vehicles[direction][lane]:
                    if v.crossed == 0 and hasattr(v, 'pcb') and v.pcb.vehicle_type == 'ambulance':
                        return i
    return None

def compute_lane_priority(i):
    if i >= len(signals): return 0
    
    congestion_part = congestion_scores[i]
    starvation_part = max_wait_time[i]
    queue_part = pcu_queued_load[i]
    
    priority_score = (0.4 * congestion_part) + (0.3 * starvation_part) + (0.3 * queue_part)
    return priority_score

def get_opf_candidate():
    # Use compute_lane_priority for selection
    scores = [(i, compute_lane_priority(i)) for i in range(noOfSignals)]
    selected_lane = max(scores, key=lambda x: x[1])[0]
    
    # Fallback to currentGreen if no valid metrics exist
    if compute_lane_priority(selected_lane) == 0:
        return currentGreen
    return selected_lane

#Signal Synchronization in repeat() and currentGreen:
def get_mlfq_candidates():
    candidates = []
    total_vehicles_waiting = 0
    with vehicle_lock:
        for direction in vehicles:
            for lane in range(3):
                total_vehicles_waiting += len([v for v in vehicles[direction][lane] if v.crossed == 0])

    # 0. Fairness Cap (PREVENT STARVATION using wait registers and max wait times)
    MAX_WAIT = 60 # Balanced from 50 to 60 for better stability
    for i in range(noOfSignals):
        if i < len(signals) and (signals[i].waitTime > MAX_WAIT or max_wait_time[i] > MAX_WAIT):
            max_val = max(signals[i].waitTime, max_wait_time[i])
            candidates.append((i, f"[FORCE] Signal {i+1} (Wait > {MAX_WAIT}s / Starvation {max_val:.1f}s)"))
            return candidates # Immediate force

    # 1. Q0 - Shockwave & Aging Promotion
    if SIMULATION_MODE == MODE_MLFQ:
        # Rapid Congestion Buildup / Shockwave promotion (highest priority MLFQ queue Q0)
        for i in range(noOfSignals):
            if i < len(signals) and queue_growth_rate[i] >= 5:
                candidates.append((i, f"[Q0 - Shockwave] Signal {i+1} (Growth Rate = {queue_growth_rate[i]:+.0f})"))
                
        # Normal aging queue promotion
        aging_lanes = []
        for i in range(noOfSignals):
            if i < len(signals) and (signals[i].waitTime >= 40 or max_wait_time[i] >= 40):
                aging_lanes.append((i, max(signals[i].waitTime, max_wait_time[i])))
        
        aging_lanes.sort(key=lambda x: x[1], reverse=True)
        for idx, wait in aging_lanes:
            candidates.append((idx, f"[Q0 - Aging] Signal {idx+1} ({wait:.1f}s)"))

    # 2. Q1 - OPF Candidates (Fairness by oldest waiting vehicle, now uses hybrid queue-intel scoring)
    opf_lane = get_opf_candidate()
    if opf_lane is not None:
        print(f"[DEBUG OPF] Selected Signal {opf_lane+1}")
        candidates.append((opf_lane, f"[Q1 - OPF] Signal {opf_lane+1} (Oldest Process)"))
    else:
        if total_vehicles_waiting > 0:
            print("[WARNING] OPF failed despite vehicles present")
        else:
            print("[DEBUG OPF] No valid candidate")

    # 3. Q2 - MVF Candidates (Throughput Priority - now Congestion-Aware)
    if SIMULATION_MODE >= MODE_MVF:
        v_counts = []
        for i in range(noOfSignals):
            if i < len(signals):
                v_counts.append((i, congestion_scores[i]))
        v_counts.sort(key=lambda x: x[1], reverse=True)
        for idx, score in v_counts:
            if score > 5.0:
                candidates.append((idx, f"[Q2 - MVF] Signal {idx+1} (Congestion Score = {score:.1f})"))

    # 4. Q3 - Round Robin (Fairness / Base Case)
    rr_start = (currentGreen + 1) % noOfSignals
    for i in range(noOfSignals):
        idx = (rr_start + i) % noOfSignals
        candidates.append((idx, f"[Q3 - RR] Signal {idx+1}"))
    return candidates


emergency_chain_count = 0

def select_next_signal():
    global emergency_chain_count, current_algorithm, last_scheduler, opf_lock_time
    
    # --- STRICT MODE CONTROL (Bypass MLFQ for Comparison) ---
    if SIMULATION_MODE == MODE_ROUND_ROBIN:
        target = (currentGreen + 1) % noOfSignals
        return {
            'type': 'NORMAL',
            'target': target,
            'source': '[RR MODE]'
        }

    if SIMULATION_MODE == MODE_MVF:
        # Most Vehicles First - Density Only
        counts = [(i, signals[i].active_count) for i in range(noOfSignals)]
        target = max(counts, key=lambda x: x[1])[0]
        return {
            'type': 'NORMAL',
            'target': target,
            'source': '[MVF MODE]'
        }

    # 0. Block during yellow or active emergency
    if is_in_yellow_phase:
        return {'type': 'HOLD', 'source': '[LOCK] Yellow Phase Active'}

    # 5. Protect OPF Execution Window
    if last_scheduler == "OPF":
        if time.time() - opf_lock_time < 3:
            return {'type': 'HOLD', 'source': '[LOCK] OPF execution'}

    # 1. Emergency Check (IRQ)
    ambQueue = checkAmbulances()
    if ambQueue:
        if is_emergency_active:
             return {'type': 'HOLD', 'source': '[LOCK] Emergency Active'}
             
        # Fix 3: Reduce Interrupt Frequency (Cooldown)
        if time.time() - last_interrupt_time < interrupt_cooldown:
            return {'type': 'HOLD', 'source': '[COOLDOWN] Interrupt blocked'}

        if emergency_chain_count < 2:
            current_algorithm = "Interrupt Handling (Emergency IRQ)"
            algo_usage[current_algorithm] += 1
            return {'type': 'EMERGENCY', 'target': ambQueue[0][0], 'source': '[INTERRUPT] Emergency'}
        else:
            log_event("SCHEDULER", "Emergency chain limit reached. Pausing new interrupts.")
            return {'type': 'HOLD', 'source': '[LIMIT] Emergency chain limit'}

    # 2. Priority Inheritance (PIP)
    pip_lane = get_pip_lane()
    if pip_lane is not None:
        if banker_resource_request(pip_lane):
            current_algorithm = "Priority Inheritance Protocol"
            algo_usage[current_algorithm] += 1
            return {
                'type': 'NORMAL',
                'target': pip_lane,
                'source': f'[PIP] Signal {pip_lane+1} (Ambulance Priority Inheritance)'
            }

    # 3. Context Restore
    if saved_context:
        current_algorithm = "Context Switching"
        algo_usage[current_algorithm] += 1
        return {'type': 'RESTORE', 'target': saved_context['index'], 'remaining': saved_context['remainingTime'], 'source': '[CONTEXT] Restore'}

    # 3. MLFQ + Banker's Gatekeeper
    candidates = get_mlfq_candidates()
    for idx, source in candidates:
        # Check Banker's safety check once
        is_granted = banker_resource_request(idx)
            
        # Fix 1: Relax Banker for OPF (Soft Override)
        if not is_granted and "Q1 - OPF" in source:
            print("[BANKER] Soft override for OPF fairness")
            is_granted = True

        if is_granted:
            # Fix 5 & 6: Track OPF execution and log
            if "Q1 - OPF" in source:
                last_scheduler = "OPF"
                opf_lock_time = time.time()
                print(f"[OPF ACTIVE] Signal {idx+1} running")
            else:
                last_scheduler = "Normal"

            if "Q0 - Shockwave" in source: current_algorithm = "MLFQ (Shockwave Promotion)"
            elif "Q0" in source: current_algorithm = "MLFQ (Aging)"
            elif "Q1 - OPF" in source: current_algorithm = "MLFQ (Oldest Process First)"
            elif "Q2" in source: current_algorithm = "MLFQ (MVF)"
            else: current_algorithm = "MLFQ (Round Robin)"
            
            algo_usage[current_algorithm] += 1

            return {'type': 'NORMAL', 'target': idx, 'source': f'--- KERNEL SCHEDULER: {source} ---'}
            
    # Banker's Fallback: Skip instead of HOLD forever if queue is very long
    for idx, source in candidates:
        with data_lock:
            if signals[idx].active_count > 15:
                log_event("BANKER", f"Unsafe state detected but Signal {idx+1} is critical. Forcing pass.")
                current_algorithm = "Banker's Override"
                return {'type': 'NORMAL', 'target': idx, 'source': f'--- KERNEL SCHEDULER (OVERRIDE): {source} ---'}

    current_algorithm = "Banker's Algorithm (Safety Check)"
    return {'type': 'HOLD', 'source': '[BANKER] Safe Hold'}

def repeat():
    global currentGreen, currentYellow, nextGreen, saved_context, last_interrupt_time, is_in_yellow_phase, is_emergency_active, emergency_chain_count, last_movement_time, waiting_for_cv, emergency_interrupt, preemption_durations
    while True:
        if waiting_for_cv:
            time.sleep(0.5)
            continue
        # Watchdog Check: Prevent false triggers during transitions
        if not is_emergency_active and not is_in_yellow_phase and signals[currentGreen].green > 0 and (time.time() - last_movement_time) > WATCHDOG_TIMEOUT:
            log_event("WATCHDOG", "No movement detected for too long. Resetting signal.")
            with data_lock:
                last_movement_time = time.time() # Prevent immediate re-trigger
            # Force transition to yellow
            if not is_in_yellow_phase:
                is_in_yellow_phase = True
                signals[currentGreen].green = 0
                signals[currentGreen].yellow = defaultYellow
                currentYellow = 1

        task = select_next_signal()
        targetSignal = task['target'] if 'target' in task else currentGreen
        
        if task['type'] != 'HOLD':
            global last_print_time
            last_print_time = 0 # Force status print on state change
        
        # --- Handle Emergency Task ---
        if task['type'] == 'EMERGENCY':
            arrival_time = time.time()
            log_event("INTERRUPT", f"Emergency on Signal {targetSignal + 1}!")
            is_emergency_active = True
            emergency_chain_count += 1
            
            # Pre-Interrupt Context Save
            if currentGreen != targetSignal and (signals[currentGreen].green > 0 or currentYellow == 1):
                if not is_in_yellow_phase:
                    saved_context = {'index': currentGreen, 'remainingTime': max(1, signals[currentGreen].green), 'timestamp': time.time()}
                    log_event("CONTEXT", f"Signal {currentGreen + 1} paused ({saved_context['remainingTime']}s)")
                    
                    is_in_yellow_phase = True
                    signals[currentGreen].green = 0
                    signals[currentGreen].yellow = defaultYellow
                    currentYellow = 1
                    while signals[currentGreen].yellow > 0:
                        printStatus(); updateValues(); time.sleep(1)
                    currentYellow = 0; signals[currentGreen].yellow = 0; signals[currentGreen].red = defaultRed; is_in_yellow_phase = False

            # Emergency Convoy Loop
            currentGreen = targetSignal; currentYellow = 0
            for i in range(noOfSignals):
                if i == currentGreen: signals[i].green = 10; signals[i].yellow = 0; signals[i].red = 0
                else: signals[i].red = defaultRed; signals[i].green = 0; signals[i].yellow = 0
            
            with data_lock:
                metrics['emergency_response_times'].append(time.time() - arrival_time)
            
            start_emergency = time.time()
            while True:
                with data_lock:
                    currentLaneAmbulances = [(idx, v) for idx, v in ambulance_queue if idx == currentGreen and v.crossed == 0]
                
                print(f"[DEBUG] Ambulances in lane: {len(currentLaneAmbulances)}")
                
                emergency_timeout = (time.time() - start_emergency) > max_emergency_duration
                # Fix 4: Emergency Timeout only if more ambulances waiting
                if len(currentLaneAmbulances) == 0 or (emergency_timeout and len(ambulance_queue) > 0):
                    if len(currentLaneAmbulances) == 0:
                        log_event("EMERGENCY", f"Signal {targetSignal + 1} cleared convoy.")
                    else:
                        log_event("WARNING", f"Emergency timeout reached for Signal {targetSignal + 1}.")
                    break 
                
                # Keep signal green
                signals[currentGreen].green = 5
                printStatus(); updateValues(); time.sleep(1)
            
            is_emergency_active = False
            last_interrupt_time = time.time()
            
            # Record emergency metrics and clear interrupt states (Phase 5 & 7)
            preemption_duration = time.time() - emergency_start_time
            preemption_durations.append(preemption_duration)
            emergency_interrupt = False
            pip_logged_directions.clear()
            
            with data_lock:
                ambulance_queue.clear() # Reset state properly
            # Final Yellow
            is_in_yellow_phase = True
            signals[currentGreen].green = 0; signals[currentGreen].yellow = defaultYellow; currentYellow = 1
            while signals[currentGreen].yellow > 0:
                printStatus(); updateValues(); time.sleep(1)
            currentYellow = 0; signals[currentGreen].yellow = 0; signals[currentGreen].red = defaultRed; is_in_yellow_phase = False
            continue

        # --- Handle Restore Task ---
        elif task['type'] == 'RESTORE':
            log_event("IRQ", "Emergency Cleared")
            log_event("CONTEXT RESTORE", f"Resuming previous scheduler state for Signal {targetSignal + 1} ({task['remaining']}s) (OS Context Switch)")
            currentGreen = targetSignal; signals[currentGreen].green = task['remaining']; saved_context = None
            emergency_chain_count = 0 # Reset chain after a successful restore or normal cycle

        # --- Handle Normal MLFQ Task ---
        elif task['type'] == 'NORMAL':
            log_event("SCHEDULER", task['source'])
            currentGreen = targetSignal; currentYellow = 0; signals[currentGreen].waitTime = 0
            setTime() 
            emergency_chain_count = 0

        # --- Handle Hold Task ---
        else:
            log_event("BANKER", "Intersection Busy - Holding")
            with data_lock:
                metrics['safety_violations_prevented'] += 1
            time.sleep(1); continue

        # --- Execution Critical Section ---
        assert targetSignal < len(signals)
        log_event("STATE", f"GREEN TS {targetSignal + 1} -> g:{signals[targetSignal].green}")
        for i in range(noOfSignals):
            if i >= len(signals): continue # Safety Check
            if i == targetSignal: signals[i].yellow = 0; signals[i].red = 0
            else: signals[i].red = defaultRed; signals[i].green = 0; signals[i].yellow = 0
        
        elapsed = 0
        initial_green = signals[currentGreen].green
        min_exec = min_execution_time
        
        while signals[currentGreen].green > 0:
            printStatus(); updateValues(); time.sleep(1); elapsed += 1
            # Preemption Check
            if (time.time() - last_interrupt_time) > interrupt_cooldown and elapsed >= min_exec:
                preempts = checkAmbulances()
                if any(idx != currentGreen and amb.crossed == 0 for idx, amb in preempts):
                    log_event("PREEMPT", "Breaking for Emergency")
                    break 

        is_in_yellow_phase = True; currentYellow = 1; signals[currentGreen].green = 0; signals[currentGreen].yellow = defaultYellow
        while signals[currentGreen].yellow > 0:
            printStatus(); updateValues(); time.sleep(1)
        currentYellow = 0; signals[currentGreen].yellow = 0; signals[currentGreen].red = defaultRed; is_in_yellow_phase = False

last_print_time = 0

def printStatus():
    global last_print_time
    # Only print every 2 seconds OR if in emergency mode
    if (time.time() - last_print_time) < 2 and not is_emergency_active:
        return
    last_print_time = time.time()
    
    for i in range(noOfSignals):
        if i >= len(signals): continue # Safety Check
        status = ""
        if i == currentGreen:
            if currentYellow == 0:
                status = " GREEN"
            else:
                status = "YELLOW"
        else:
            if signals[i].yellow > 0:
                status = "YELLOW"
            else:
                status = "   RED"
        
        with data_lock:
            if i < len(signals):
                g_display = signals[i].green
                print(f"{status} TS {i+1} -> r:{signals[i].red} y:{signals[i].yellow} g:{g_display} | Active:{signals[i].active_count} S:{signals[i].spawned_count} E:{signals[i].exited_count}")
            else:
                log_event("FIXED", f"Skipped OOB status print for Index {i}")
    print()

def updateValues():
    if waiting_for_cv:
        return
    
    global queue_length, stopped_vehicles_count, slow_vehicles_count
    global average_wait_time, max_wait_time, queue_growth_rate, previous_queue_length, congestion_scores
    global queue_metrics_history
    global pcu_load, pcu_queued_load, pcu_metrics_history

    # 2. Save previous queue lengths for growth rate calculation (shockwave analysis)
    for i in range(noOfSignals):
        previous_queue_length[i] = queue_length[i]

    # 3. Update Queue Intelligence and PCU metrics for all 4 approach corridors
    for i in range(noOfSignals):
        direction = directionNumbers[i]
        total_active_vehicles = 0
        q_len = 0
        stopped_cnt = 0
        slow_cnt = 0
        total_wait = 0.0
        max_wait = 0.0
        
        p_load = 0.0
        p_q_load = 0.0
        
        is_red = (currentGreen != i)
        stop_line = stopLines[direction]
        
        with vehicle_lock:
            for lane in range(3):
                for v in vehicles[direction][lane]:
                    if v.crossed == 0:
                        total_active_vehicles += 1
                        v_class = v.vehicleClass
                        v_pcu = vehicle_pcu_weights.get(v_class, 1.0)
                        p_load += v_pcu
                        
                        # Distance to signal stop line (pixels)
                        if direction == 'right':
                            distance_to_stop = stop_line - (v.x + v.currentImage.get_rect().width)
                        elif direction == 'left':
                            distance_to_stop = v.x - stop_line
                        elif direction == 'down':
                            distance_to_stop = stop_line - (v.y + v.currentImage.get_rect().height)
                        elif direction == 'up':
                            distance_to_stop = v.y - stop_line
                            
                        # Queue detection logic: speed under threshold OR waiting near red signal
                        v_moved = getattr(v, 'moved_distance', v.speed)
                        
                        v_stopped = (v_moved < 0.1)
                        v_slow = (0.1 <= v_moved < 1.2)
                        v_queued = (v_moved < 1.2) or (is_red and distance_to_stop < 150)
                        
                        # Update state registers in Vehicle PCB
                        v.pcb.is_stopped = v_stopped
                        v.pcb.is_slow = v_slow
                        v.pcb.is_queued = v_queued
                        
                        # Accumulate wait times inside queue segments
                        if v_queued:
                            if v.pcb.queue_start_time is None:
                                v.pcb.queue_start_time = time.time()
                        else:
                            if v.pcb.queue_start_time is not None:
                                v.pcb.accumulated_wait_time += (time.time() - v.pcb.queue_start_time)
                                v.pcb.queue_start_time = None
                                
                        v_wait = v.pcb.get_current_wait_time()
                        total_wait += v_wait
                        if v_wait > max_wait:
                            max_wait = v_wait
                            
                        if v_queued:
                            q_len += 1
                            p_q_load += v_pcu
                        if v_stopped:
                            stopped_cnt += 1
                        if v_slow:
                            slow_cnt += 1
                            
        # Commit updated lane state to global registers
        queue_length[i] = q_len
        pcu_load[i] = p_load
        pcu_queued_load[i] = p_q_load
        stopped_vehicles_count[i] = stopped_cnt
        slow_vehicles_count[i] = slow_cnt
        average_wait_time[i] = total_wait / total_active_vehicles if total_active_vehicles > 0 else 0.0
        max_wait_time[i] = max_wait
        
        # Calculate queue growth rate (differential index)
        queue_growth_rate[i] = q_len - previous_queue_length[i]
        
        # Congestion Score Engine (Hybrid Congestion Score equation, upgraded to evaluate queued PCU load)
        congestion_scores[i] = max(0.0, ALPHA_QUEUE * pcu_queued_load[i] + BETA_WAIT * average_wait_time[i] + GAMMA_GROWTH * queue_growth_rate[i])
        
        # Shockwave Propagation detection
        if queue_growth_rate[i] >= 5:
            log_event("SHOCKWAVE", f"Rapid congestion buildup detected in Lane {i+1}")

    # Log queue average wait times periodically (every 2 seconds)
    if timeElapsed % 2 == 0:
        for i in range(noOfSignals):
            log_event("QUEUE", f"Lane {i+1} avg wait = {average_wait_time[i]:.1f}s")

    # Record telemetry for CSV dataset compilation
    with data_lock:
        queue_metrics_history.append({
            'time': timeElapsed,
            'queue_lengths': list(queue_length),
            'avg_wait_times': list(average_wait_time),
            'max_wait_times': list(max_wait_time),
            'growth_rates': list(queue_growth_rate),
            'congestion_scores': list(congestion_scores)
        })
        pcu_metrics_history.append({
            'time': timeElapsed,
            'pcu_loads': list(pcu_load),
            'pcu_queued_loads': list(pcu_queued_load),
            'congestion_scores': list(congestion_scores),
            'static_pcu_weights': dict(vehicle_pcu_weights)
        })

    # Banker's Resource Tracking: Update global intersection occupancy and matrices
    update_banker_state()
    
    # Phase 4: Priority Inheritance Update
    apply_priority_inheritance()
    
    for i in range(noOfSignals):
        with data_lock:
            if i < len(signals):
                # Safety Check: Explicit Accounting Integrity
                expected_active = signals[i].spawned_count - signals[i].exited_count
                if signals[i].active_count != expected_active:
                    log_event("FIXED", f"Count correction Signal {i+1}: Active({signals[i].active_count}) != Expected({expected_active})")
                    signals[i].active_count = max(0, expected_active) # Auto-correction
                
                if signals[i].active_count < 0:
                    log_event("FIXED", f"Negative active_count reset on Signal {i+1}")
                    signals[i].active_count = 0

                if i == currentGreen:
                    if currentYellow == 0:
                        if signals[i].green > 0:
                            signals[i].green -= 1
                            signals[i].totalGreenTime += 1
                        signals[i].waitTime = 0  # Reset wait time while green
                    else:
                        if signals[i].yellow > 0:
                            signals[i].yellow -= 1
                else:
                    # Increment wait time for red signals
                    signals[i].waitTime += 1
                    if signals[i].yellow > 0:
                        signals[i].yellow -= 1
                    elif signals[i].red > 0:
                        signals[i].red -= 1
            else:
                log_event("FIXED", f"Skipped OOB update for Index {i}")

        # --- Queue Advancement Logic (Dynamic Stop Lines) ---
        direction = directionNumbers[i]
        with vehicle_lock:
            for lane in range(3):
                lane_vehicles = vehicles[direction][lane]
                first_not_crossed_idx = -1
                for j in range(len(lane_vehicles)):
                    if lane_vehicles[j].crossed == 0:
                        first_not_crossed_idx = j
                        break
                
                if first_not_crossed_idx != -1:
                    # Update stop for the lead vehicle
                    lane_vehicles[first_not_crossed_idx].stop = defaultStop.get(direction, 500)
                    # Update stop for all following vehicles sequentially
                    for j in range(first_not_crossed_idx + 1, len(lane_vehicles)):
                        prev = lane_vehicles[j-1]
                        curr = lane_vehicles[j]
                        if direction == 'right':
                            curr.stop = prev.stop - prev.currentImage.get_rect().width - gap
                        elif direction == 'left':
                            curr.stop = prev.stop + prev.currentImage.get_rect().width + gap
                        elif direction == 'down':
                            curr.stop = prev.stop - prev.currentImage.get_rect().height - gap
                        elif direction == 'up':
                            curr.stop = prev.stop + prev.currentImage.get_rect().height + gap

# --- CV CONFIGURATION & MODULAR HELPERS (SCENARIO 2 HYBRID SENSOR) ---
USE_CV_DATA = True
CV_MULTIPLIER = 1 # Set to 1 for perfect 1-to-1 Digital Twin
MAX_SPAWN_QUEUE = 50
DEBUG_CV = True

# Global Sparse Sensor Telemetry States
live_density_index = 1.0
measured_count_ns = 0
synthetic_spawn_east = 0
synthetic_spawn_west = 0
total_intersection_load = 0

# --- HISTORICAL CORRIDOR BASELINES FROM DATASET ---
v0_mean = 10000.0
v1_mean = 10000.0
v2_mean = 10000.0
v3_mean = 10000.0

if df is not None:
    try:
        v0_mean = df[df['Road/Intersection Name'] == "100 Feet Road"]['Traffic Volume'].mean()
        v1_mean = df[df['Road/Intersection Name'] == "CMH Road"]['Traffic Volume'].mean()
        v2_mean = df[df['Road/Intersection Name'] == "Marathahalli Bridge"]['Traffic Volume'].mean()
        v3_mean = df[df['Road/Intersection Name'] == "Sony World Junction"]['Traffic Volume'].mean()
        
        # Handle potential NaNs cleanly
        v0_mean = v0_mean if not np.isnan(v0_mean) else 10000.0
        v1_mean = v1_mean if not np.isnan(v1_mean) else 10000.0
        v2_mean = v2_mean if not np.isnan(v2_mean) else 10000.0
        v3_mean = v3_mean if not np.isnan(v3_mean) else 10000.0
    except Exception as e:
        print(f"[ERROR] Historical baseline mean calculation failed: {e}")

# Normalize to 1-second arrival rate baselines (Traffic Volume is hourly in typical datasets)
avg_arrival_rate_ns = (v0_mean + v2_mean) / 3600.0
avg_arrival_rate_east = v1_mean / 3600.0
avg_arrival_rate_west = v3_mean / 3600.0

# Ensure baseline is not zero to avoid division by zero
if avg_arrival_rate_ns <= 0:
    avg_arrival_rate_ns = 0.5
if avg_arrival_rate_east <= 0:
    avg_arrival_rate_east = 0.25
if avg_arrival_rate_west <= 0:
    avg_arrival_rate_west = 0.25

print(f"[SPARSE SENSOR KERNEL] Calculated 1-second Historical Baselines:")
print(f"  N/S Monitored Corridor (100 Feet + Marathahalli): {avg_arrival_rate_ns:.4f} vehicles/sec")
print(f"  East Corridor (CMH Road): {avg_arrival_rate_east:.4f} vehicles/sec")
print(f"  West Corridor (Sony World): {avg_arrival_rate_west:.4f} vehicles/sec")

# lane_spawning_queues now holds deques of vehicle strings
lane_spawning_queues = [deque() for _ in range(4)]
cv_metrics = {"total_cv_spawned": 0, "read_errors": 0}

# Mapping strings back to Pygame IDs
TYPE_TO_ID = {'car': 0, 'bus': 1, 'truck': 2, 'rickshaw': 3, 'bike': 4, 'ambulance': 5}

def read_cv_data():
    """Safely reads the JSON bridge and resets it to prevent double-counting.
       Robustly handles both list-based and count-based JSON formats."""
    global waiting_for_cv
    try:
        with open("cv_traffic_counts.json", "r") as f:
            data = json.load(f)
        
        # If waiting, check if any lane count is a list (indicates cv_sensor.py has started writing)
        if waiting_for_cv:
            is_valid_sensor_data = False
            for i in range(4):
                val = data.get(f"lane_{i}")
                if isinstance(val, list):
                    is_valid_sensor_data = True
                    break
            if is_valid_sensor_data:
                waiting_for_cv = False
                log_event("SUCCESS", "CV Sensor connected! Resuming simulation.")
        
        # Check if any lane has new arrivals
        has_arrivals = False
        for i in range(4):
            val = data.get(f"lane_{i}", [])
            if isinstance(val, list) and len(val) > 0:
                has_arrivals = True
                break
                
        if has_arrivals:
            # Reset the bridge file to empty lists to prevent double-counting
            with open("cv_traffic_counts.json", "w") as f:
                json.dump({"lane_0": [], "lane_1": [], "lane_2": [], "lane_3": []}, f)
            return data
    except Exception as e:
        cv_metrics["read_errors"] += 1
    return None

def update_spawn_queue(arrivals):
    """Calculates live density index using monitored lanes 0 & 2,
       and dynamically scales synthetic lane 1 & 3 spawning rates
       based on Bangalore historical traffic datasets (Scenario 2)."""
    global live_density_index, measured_count_ns, synthetic_spawn_east, synthetic_spawn_west, total_intersection_load
    
    # 1. Monitored Lane Detections (REAL)
    live_nb = arrivals.get("lane_0", [])
    live_sb = arrivals.get("lane_2", [])
    measured_count_ns = len(live_nb) + len(live_sb)
    
    # 2. Live Density Estimation (Phase 3)
    # Estimate the multiplier comparing current arrivals in 1 sec to average N/S arrivals
    live_density_index = float(measured_count_ns) / avg_arrival_rate_ns
    live_density_index = max(0.1, min(3.0, live_density_index)) # Clamp to [0.1, 3.0]
    
    # 3. Queue Measured Spawns (lane_0 and lane_2)
    for v_type in live_nb:
        if len(lane_spawning_queues[0]) < MAX_SPAWN_QUEUE:
            lane_spawning_queues[0].append(v_type)
            
    for v_type in live_sb:
        if len(lane_spawning_queues[2]) < MAX_SPAWN_QUEUE:
            lane_spawning_queues[2].append(v_type)
            
    # 4. Generate Calibrated Synthetic Spawns (lane_1 and lane_3) (Phase 4)
    # Calculate target counts using mathematical calibration
    spawn_east_rate = avg_arrival_rate_east * live_density_index
    synthetic_spawn_east = int(spawn_east_rate)
    if random.random() < (spawn_east_rate - synthetic_spawn_east):
        synthetic_spawn_east += 1
        
    spawn_west_rate = avg_arrival_rate_west * live_density_index
    synthetic_spawn_west = int(spawn_west_rate)
    if random.random() < (spawn_west_rate - synthetic_spawn_west):
        synthetic_spawn_west += 1
        
    # Bangalore traffic distribution probabilities (car: 50%, bike: 35%, bus: 5%, truck: 5%, rickshaw: 5%)
    choices = ['car', 'bike', 'bus', 'truck', 'rickshaw']
    weights = [0.50, 0.35, 0.05, 0.05, 0.05]
    
    for _ in range(synthetic_spawn_east):
        if len(lane_spawning_queues[1]) < MAX_SPAWN_QUEUE:
            v_type_str = np.random.choice(choices, p=weights)
            lane_spawning_queues[1].append(v_type_str)
            
    for _ in range(synthetic_spawn_west):
        if len(lane_spawning_queues[3]) < MAX_SPAWN_QUEUE:
            v_type_str = np.random.choice(choices, p=weights)
            lane_spawning_queues[3].append(v_type_str)
            
    # Total Calibrated Intersection Load Tracker
    total_intersection_load = len(lane_spawning_queues[0]) + len(lane_spawning_queues[1]) + len(lane_spawning_queues[2]) + len(lane_spawning_queues[3])
    
    # Print Cycle Log in Terminal (Phase 6)
    print(f"\n[SPARSE-SENSOR LOG] Sync Cycle Update:")
    print(f"  Live Detections -> Northbound: {len(live_nb)} | Southbound: {len(live_sb)} (Total Measured: {measured_count_ns})")
    print(f"  Live Density Index (lambda): {live_density_index:.4f}")
    print(f"  Calibrated Spawns -> East (CMH Road): {synthetic_spawn_east} | West (Sony World): {synthetic_spawn_west}")
    print(f"  Active Simulation Queue Load: {total_intersection_load} vehicles in system")

def spawn_from_queue():
    """Pops one vehicle per lane per tick to ensure smooth Pygame rendering."""
    for dir_num in range(4):
        if lane_spawning_queues[dir_num]:
            v_class_str = lane_spawning_queues[dir_num].popleft()
            v_type_id = TYPE_TO_ID.get(v_class_str, 0)
            
            # Logic for lane selection (Bike/Lane 0, others/Lane 1-2)
            lane_idx = 0 if v_type_id == 4 else random.randint(1, 2)
            will_turn = 1 if lane_idx == 2 and random.randint(0, 4) <= 2 else 0
            
            Vehicle(lane_idx, v_class_str, dir_num, directionNumbers[dir_num], will_turn)
            cv_metrics["total_cv_spawned"] += 1

def generateVehicles():
    """Main Threaded Spawning Loop"""
    global waiting_for_cv, last_ambulance_spawn_time
    # Safe init of JSON file
    if USE_CV_DATA:
        try:
            with open("cv_traffic_counts.json", "w") as f:
                json.dump({"lane_0": [], "lane_1": [], "lane_2": [], "lane_3": []}, f)
        except: pass
    else:
        waiting_for_cv = False

    while True:
        if USE_CV_DATA:
            arrivals = read_cv_data()
            if arrivals:
                update_spawn_queue(arrivals)
            
            # Random Ambulance Spawning (Phase 1)
            # Spawns emergency vehicles in any of the 4 lanes under low probability and cooldown
            if time.time() - last_ambulance_spawn_time > emergency_cooldown:
                # low probability of ambulance spawning (0.6% chance per tick)
                if random.random() < 0.006:
                    dir_num = random.choice([0, 1, 2, 3])
                    # Middle lane (lane 1) for ambulance
                    lane_idx = 1
                    Vehicle(lane_idx, 'ambulance', dir_num, directionNumbers[dir_num], 0)
                    last_ambulance_spawn_time = time.time()
                    
            spawn_from_queue()
            time.sleep(0.2) # Thread-safe tick
            continue
            
        # --- ORIGINAL RANDOM SPAWNING LOGIC (FALLBACK) ---
        vehicle_type = random.randint(0, 5)
        if vehicle_type == 5:  # Ambulance
            if time.time() - last_ambulance_spawn_time > emergency_cooldown and len(ambulance_queue) == 0:
                direction_number = random.choice([0, 1, 2, 3])
                # Middle lane (lane 1) for ambulance
                lane_idx = 1
                Vehicle(lane_idx, 'ambulance', direction_number, directionNumbers[direction_number], 0)
                last_ambulance_spawn_time = time.time()
            continue
        else:
            if vehicle_type == 4:
                lane_number = 0
            else:
                lane_number = random.randint(0, 1) + 1
            will_turn = 0
            if lane_number == 2:
                temp = random.randint(0, 4)
                if temp <= 2:
                    will_turn = 1
            temp = random.randint(0, 999)
            direction_number = 0
            a = [400, 800, 900, 1000]
            if temp < a[0]:
                direction_number = 0
            elif temp < a[1]:
                direction_number = 1
            elif temp < a[2]:
                direction_number = 2
            elif temp < a[3]:
                direction_number = 3
            Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        
        # Predictive Spawning: Adjust sleep interval based on Dataset Growth Factor
        base_sleep = 0.5 # Increased spawn rate for 70s simulation
        scaled_sleep = max(0.1, base_sleep / growth_factor)
        time.sleep(scaled_sleep)


def simulationTime():
    global timeElapsed, simTime, lane_load_data, lane_wait_times, waiting_for_cv
    while True:
        if waiting_for_cv:
            time.sleep(0.5)
            continue
        timeElapsed += 1
        time_data.append(timeElapsed)
        throughput_data.append(metrics['total_vehicles_cleared'])
        
        # Safe Metrics Update & Validation
        with data_lock:
            if len(lane_load_data) != noOfSignals:
                lane_load_data = [[] for _ in range(noOfSignals)]
                log_event("FIXED", "Re-initialized lane_load_data due to mismatch")
            
            if len(lane_wait_times) != noOfSignals:
                lane_wait_times = [[] for _ in range(noOfSignals)]
                log_event("FIXED", "Re-initialized lane_wait_times due to mismatch")

        for i in range(noOfSignals):
            # Safe Signal Access
            with data_lock:
                if i < len(signals) and i < len(lane_load_data):
                    lane_load_data[i].append(signals[i].active_count)
                    if timeElapsed % 30 == 0:
                        log_event("FIXED", f"Safe metrics update for Index {i}")
                else:
                    if timeElapsed % 10 == 0:
                        log_event("FIXED", f"Skipped OOB access for Index {i}")
            
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = metrics['total_vehicles_cleared']
            print('Lane-wise Vehicle Counts')
            output_lines = ['Lane-wise Vehicle Counts']
            for i in range(noOfSignals):
                lane_total = vehicles[directionNumbers[i]]['crossed']
                lane_output = f'Lane {i + 1}: {lane_total}'
                print(lane_output)
                output_lines.append(lane_output)
            
            total_output = f'Total vehicles passed: {totalVehicles}'
            time_output = f'Total time passed: {timeElapsed}'
            throughput = float(totalVehicles) / float(timeElapsed)
            throughput_output = f'No. of vehicles passed per unit time: {throughput:.2f}'
            
            with data_lock:
                avg_wait = metrics['total_wait_time'] / totalVehicles if totalVehicles > 0 else 0
                avg_response = sum(metrics['emergency_response_times']) / len(metrics['emergency_response_times']) if metrics['emergency_response_times'] else 0
            
            metrics_output = [
                total_output, 
                time_output, 
                throughput_output,
                f'Average Wait Time: {avg_wait:.2f}s',
                f'Average Emergency Response Time: {avg_response:.2f}s'
            ]
            
            for line in metrics_output:
                print(line)
                output_lines.append(line)

            # --- Export Data for Analysis ---
            results = {
                "time_data": time_data,
                "throughput": throughput_data,
                "lane_wait_times": lane_wait_times,
                "lane_load": lane_load_data,
                "algo_usage": algo_usage,
                "emergency_response": metrics['emergency_response_times'],
                "banker": banker_stats,
                "emergency_events": {
                    "total_events": total_emergency_events,
                    "preemption_durations": preemption_durations,
                    "interrupted_signals": interrupted_signal_stats
                }
            }
            
            mode_name = ["rr", "mvf", "mlfq"][SIMULATION_MODE]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            os.makedirs("results", exist_ok=True)
            
            # --- Export Emergency Metrics to CSV (Phase 7) ---
            try:
                csv_file = os.path.join("results", f"emergency_metrics_seed{SEED}_{timestamp}.csv")
                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Event ID", "Interrupted Signal", "Remaining Green (s)", "Response Latency (s)", "Clearance Time (s)"])
                    for idx in range(total_emergency_events):
                        interrupted_sig = interrupted_signal_stats[idx]['interrupted_signal'] if idx < len(interrupted_signal_stats) else "N/A"
                        rem_green = interrupted_signal_stats[idx]['remaining_green'] if idx < len(interrupted_signal_stats) else "N/A"
                        latency = metrics['emergency_response_times'][idx] if idx < len(metrics['emergency_response_times']) else "N/A"
                        duration = preemption_durations[idx] if idx < len(preemption_durations) else "N/A"
                        writer.writerow([idx + 1, interrupted_sig, rem_green, f"{latency:.2f}" if isinstance(latency, float) else latency, f"{duration:.2f}" if isinstance(duration, float) else duration])
                log_event("SUCCESS", f"Emergency metrics exported to {csv_file}")
            except Exception as e:
                log_event("ERROR", f"Emergency metrics CSV export failed: {e}")

            # --- Export Queue Intelligence Metrics to CSV (Step 7) ---
            try:
                queue_csv_file = os.path.join("results", f"queue_metrics_seed{SEED}_{timestamp}.csv")
                with open(queue_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Compile header row
                    header = ["Time"]
                    for lane in range(1, 5):
                        header.extend([
                            f"L{lane}_QueueLength",
                            f"L{lane}_AvgWaitTime",
                            f"L{lane}_MaxWaitTime",
                            f"L{lane}_GrowthRate",
                            f"L{lane}_CongestionScore"
                        ])
                    writer.writerow(header)
                    # Compile data rows
                    for record in queue_metrics_history:
                        row = [record['time']]
                        for i in range(4):
                            row.extend([
                                record['queue_lengths'][i],
                                f"{record['avg_wait_times'][i]:.2f}",
                                f"{record['max_wait_times'][i]:.2f}",
                                record['growth_rates'][i],
                                f"{record['congestion_scores'][i]:.2f}"
                            ])
                        writer.writerow(row)
                log_event("SUCCESS", f"Queue intelligence metrics exported to {queue_csv_file}")
            except Exception as e:
                log_event("ERROR", f"Queue metrics CSV export failed: {e}")

            # --- Export PCU Intelligence Metrics to CSV (Phase 2 Step 8) ---
            try:
                pcu_csv_file = os.path.join("results", f"pcu_metrics_seed{SEED}_{timestamp}.csv")
                with open(pcu_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Compile header row
                    header = ["Time"]
                    for lane in range(1, 5):
                        header.extend([
                            f"L{lane}_ActivePCULoad",
                            f"L{lane}_QueuedPCULoad",
                            f"L{lane}_CongestionScore"
                        ])
                    for c in sorted(speeds.keys()):
                        header.append(f"PCU_Weight_{c}")
                    writer.writerow(header)
                    # Compile data rows
                    for record in pcu_metrics_history:
                        row = [record['time']]
                        for i in range(4):
                            row.extend([
                                f"{record['pcu_loads'][i]:.2f}",
                                f"{record['pcu_queued_loads'][i]:.2f}",
                                f"{record['congestion_scores'][i]:.2f}"
                            ])
                        for c in sorted(speeds.keys()):
                            row.append(f"{record['static_pcu_weights'][c]:.2f}")
                        writer.writerow(row)
                log_event("SUCCESS", f"PCU metrics exported to {pcu_csv_file}")
            except Exception as e:
                log_event("ERROR", f"PCU metrics CSV export failed: {e}")

            # Export queue/pcu data inside results JSON as well
            results["queue_history"] = queue_metrics_history
            results["pcu_history"] = pcu_metrics_history

            if MODEL_NAME:

                filename = os.path.join("results", f"results_{MODEL_NAME.lower()}_{mode_name}_t{simTime}_seed{SEED}_{timestamp}.json")
            else:
                filename = os.path.join("results", f"results_{mode_name}_t{simTime}_seed{SEED}_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump(results, f)
            log_event("SUCCESS", f"Simulation data exported to {filename} (Duration: {simTime}s)")

            # --- Final Performance Log (Architectural Alignment) ---
            output_dir = os.path.join(os.getcwd(), "Outputs")
            try:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(output_dir, f'architectural_audit_{timestamp}.txt')
                with open(log_file, 'w') as f:
                    f.write('\n'.join(output_lines))
                log_event("SUCCESS", f"Final Audit Report Generated: {log_file}")
            except Exception as e:
                log_event("ERROR", f"Reporting failure: {e}")

            # --- Critical for Experiments: Exit cleanly after completion ---
            print("\n[FINISH] Simulation Cycle Complete. Exiting process.")
            os._exit(0)
# The code simulates system calls by performing file I/O (open, write) and process management (via threads)
def Main():
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)

    simulationWidth = 1400
    screenWidth = simulationWidth
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)

    pygame.display.set_caption("SIMULATION")

    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            # --- MANUAL EMERGENCY TRIGGER FOR DEMO ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    # Spawn emergency vehicle in a random direction
                    dir_num = random.randint(0, 3)
                    print(f"\n[IRQ] MANUAL EMERGENCY TRIGGER: Spawning Ambulance in Lane {dir_num+1}")
                    Vehicle(1, 'ambulance', dir_num, directionNumbers[dir_num], 0)

        screen.blit(background, (0, 0))
        
        # --- Translucent Emergency Lane Glow (Phase 3 & 6) ---
        # Analogy: Drawing focus/priority highlight on active interrupt source
        if is_emergency_active:
            # Alternate colors every 200ms
            flash_color = (255, 0, 0) if (int(time.time() * 5) % 2 == 0) else (0, 0, 255)
            glow_surf = pygame.Surface((screenWidth, screenHeight), pygame.SRCALPHA)
            
            # Draw overlay rectangle with 70 alpha (translucent) on the active lane corridor
            if active_emergency_lane == 0: # Right direction
                pygame.draw.rect(glow_surf, (*flash_color, 70), (0, 340, 590, 70))
            elif active_emergency_lane == 1: # Down direction
                pygame.draw.rect(glow_surf, (*flash_color, 70), (680, 0, 70, 330))
            elif active_emergency_lane == 2: # Left direction
                pygame.draw.rect(glow_surf, (*flash_color, 70), (800, 430, 600, 70))
            elif active_emergency_lane == 3: # Up direction
                pygame.draw.rect(glow_surf, (*flash_color, 70), (600, 535, 70, 265))
                
            screen.blit(glow_surf, (0, 0))

        # --- Congestion Lane Color Intensity Overlay (Step 6) ---
        cong_surf = pygame.Surface((screenWidth, screenHeight), pygame.SRCALPHA)
        for i in range(noOfSignals):
            score = congestion_scores[i]
            if score > 0.5:
                # Alpha opacity proportional to congestion score (max 140/255)
                alpha = min(140, int(score * 3.0))
                color = (255, 0, 0, alpha)
                if i == 0: # Right (Lane 1 approach)
                    pygame.draw.rect(cong_surf, color, (0, 340, 590, 70))
                elif i == 1: # Down (Lane 2 approach)
                    pygame.draw.rect(cong_surf, color, (680, 0, 70, 330))
                elif i == 2: # Left (Lane 3 approach)
                    pygame.draw.rect(cong_surf, color, (800, 430, 600, 70))
                elif i == 3: # Up (Lane 4 approach)
                    pygame.draw.rect(cong_surf, color, (600, 535, 70, 265))
        screen.blit(cong_surf, (0, 0))


        for i in range(noOfSignals):
            if i >= len(signals): continue # Safety Check
            if i == currentGreen:
                if currentYellow == 1:
                    if signals[i].yellow == 0:
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if signals[i].green == 0:
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].yellow > 0:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                elif signals[i].red <= 10:
                    if signals[i].red == 0:
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                    screen.blit(redSignal, signalCoods[i])
                else:
                    signals[i].signalText = "---"
                    screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        for i in range(noOfSignals):
            if i >= len(signals): continue # Safety Check
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])
            
            # Draw REAL / SYNTH labels near the counts (Phase 6 HUD)
            # i=0 and i=2 are measured via YOLO, i=1 and i=3 are synthetically calibrated
            indicator_coods = [(480, 185), (880, 185), (880, 580), (480, 580)]
            if i in [0, 2]:
                ind_label = font.render("REAL", True, (0, 255, 0), black)
            else:
                ind_label = font.render("SYNTH", True, (255, 255, 0), black)
            screen.blit(ind_label, indicator_coods[i])

        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, (1100, 50))
        
        # --- Kernel HUD Overlay (Phase 4 Cleanup & Scenario 2 Calibration HUD) ---
        hud_font = pygame.font.Font(None, 28)
        mode_text = ["Round-Robin", "MVF (Density Only)", "Full OS MLFQ"][SIMULATION_MODE]
        mode_label = hud_font.render(f"KERNEL MODE: {mode_text}", True, (255, 255, 0), black)
        screen.blit(mode_label, (50, 50))
        
        # Display Current Algorithm
        algo_label = hud_font.render(f"ALGORITHM: {current_algorithm}", True, (0, 255, 255), black)
        screen.blit(algo_label, (50, 80))
        
        # Display Current State
        state_text = "STATE: IDLE"
        if waiting_for_cv:
            state_text = "STATE: WAITING FOR CV SENSOR BRIDGING..."
        elif is_emergency_active:
            state_text = f"STATE: EMERGENCY PREEMPTION ACTIVE"
        elif is_in_yellow_phase:
            state_text = f"STATE: CLEARING YELLOW (Signal {currentGreen+1})"
        else:
            state_text = f"STATE: GREEN (Time-Based: {signals[currentGreen].green}s)"
        state_label = hud_font.render(state_text, True, white, black)
        screen.blit(state_label, (50, 110))

        # Vehicles Cleared
        cleared_label = hud_font.render(f"Vehicles Cleared: {metrics['total_vehicles_cleared']}", True, white, black)
        screen.blit(cleared_label, (50, 140))

        # Sparse Sensor Digital Twin Label
        sensor_twin_label = hud_font.render("SPARSE-SENSOR DIGITAL TWIN (Scenario 2)", True, (234, 179, 8), black)
        screen.blit(sensor_twin_label, (50, 180))

        # Detections & Density Metrics
        ns_label = hud_font.render(f"Live N/S Count (YOLO): {measured_count_ns}", True, (34, 197, 94), black)
        screen.blit(ns_label, (50, 210))

        density_label = hud_font.render(f"Live Density Index (lambda): {live_density_index:.2f}", True, white, black)
        screen.blit(density_label, (50, 240))

        east_label = hud_font.render(f"Calibrated East Spawn (CMH): {synthetic_spawn_east}", True, (234, 179, 8), black)
        screen.blit(east_label, (50, 270))

        west_label = hud_font.render(f"Calibrated West Spawn (Sony): {synthetic_spawn_west}", True, (234, 179, 8), black)
        screen.blit(west_label, (50, 300))

        load_label = hud_font.render(f"Total Simulation Load: {total_intersection_load} Veh | {sum(pcu_load):.2f} PCU", True, (0, 255, 255), black)
        screen.blit(load_label, (50, 330))

        # --- STATIC PCU OCCUPANCY-AWARE SCHEDULING HUD ---
        pcu_font = pygame.font.Font(None, 20)
        pcu_intel_label = pcu_font.render("Static PCU Occupancy-Aware Scheduling Layer", True, (255, 0, 128), black)
        screen.blit(pcu_intel_label, (980, 530))
        
        for i in range(noOfSignals):
            load = pcu_load[i]
            if load < 5:
                severity = "Low"
            elif load < 15:
                severity = "Medium"
            else:
                severity = "High"
            
            lane_stats = f"PCU Load: {load:.2f} | Severity: {severity} | Congestion: {congestion_scores[i]:.1f}"
            lane_label = pcu_font.render(f"Lane {i+1}: {lane_stats}", True, (255, 255, 255), black)
            screen.blit(lane_label, (950, 575 + i * 20))

        # --- EMERGENCY WARNING BANNER (Phase 3 & 6) ---
        if is_emergency_active:
            # Flashing red background banner
            banner_bg = (180, 0, 0) if (int(time.time() * 3) % 2 == 0) else (40, 0, 0)
            pygame.draw.rect(screen, banner_bg, (1000, 95, 360, 45))
            pygame.draw.rect(screen, white, (1000, 95, 360, 45), 2)
            
            emergency_banner_font = pygame.font.Font(None, 20)
            eb_label = emergency_banner_font.render("EMERGENCY PREEMPTION ACTIVE (Hardware IRQ)", True, white)
            eb_sublabel = emergency_banner_font.render("Analogy: CPU Preemption & Stack Save/Restore", True, (255, 255, 0))
            screen.blit(eb_label, (1010, 100))
            screen.blit(eb_sublabel, (1010, 118))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
            
            # --- Flashing red/blue ambulance siren lights (Phase 6) ---
            if vehicle.vehicleClass == 'ambulance':
                # Alternate color at 6Hz frequency
                flash_color = (255, 0, 0) if (int(time.time() * 6) % 2 == 0) else (0, 0, 255)
                # Position siren light directly in center of ambulance bounding box
                cx = int(vehicle.x + vehicle.currentImage.get_rect().width // 2)
                cy = int(vehicle.y + vehicle.currentImage.get_rect().height // 2)
                pygame.draw.circle(screen, flash_color, (cx, cy), 10)
                pygame.draw.circle(screen, white, (cx, cy), 5) # Core glow
                
        pygame.display.update()

if __name__ == "__main__":
    Main()
