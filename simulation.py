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
INTERSECTION_CAPACITY = 40 
banker_available = INTERSECTION_CAPACITY
banker_allocation = [0, 0, 0, 0]
banker_need = [0, 0, 0, 0]
banker_max = [INTERSECTION_CAPACITY] * 4 

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

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 5
defaultMaximum = 20

signals = []
simTime = 100
timeElapsed = 0

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
        alloc = [0, 0, 0, 0]
        need = [0, 0, 0, 0]
        for i in range(noOfSignals):
            direction = directionNumbers[i]
            waiting = 0
            in_intersection = 0
            for lane in range(3):
                for v in vehicles[direction][lane]:
                    if v.crossed == 0:
                        waiting += 1
                    else:
                        # Check if still in intersection (boundary check)
                        if direction == 'right' and v.x < 1400: in_intersection += 1
                        elif direction == 'down' and v.y < 800: in_intersection += 1
                        elif direction == 'left' and v.x > 0: in_intersection += 1
                        elif direction == 'up' and v.y > 0: in_intersection += 1
            alloc[i] = in_intersection
            need[i] = waiting
        
        with data_lock:
            banker_allocation = alloc
            banker_need = need
            banker_available = max(0, INTERSECTION_CAPACITY - sum(alloc))

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
        if banker_allocation[signal_idx] > 0:
            return True # Avoid double allocation
            
        # Check 1: Request <= Need (Always true by definition here)
        # Check 2: Request <= Available
        if request > banker_available:
            log_event("BANKER", f"Request DENIED for Signal {signal_idx+1}: Request ({request}) > Available ({banker_available})")
            return False
            
        # Temporary Allocation
        banker_available = max(0, banker_available - request)
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
def apply_priority_inheritance():
    with vehicle_lock:
        for direction in directionNumbers.values():
            for lane in range(3):
                queue = vehicles[direction][lane]
                # Find the highest priority in the queue (e.g., an ambulance)
                max_prio = 0
                for v in queue:
                    if hasattr(v, 'pcb') and v.pcb.vehicle_type == 'ambulance' and v.crossed == 0:
                        max_prio = 10 # Critical Priority
                        break
                
                # Inherit priority to everyone in front of the ambulance
                if max_prio > 0:
                    for v in queue:
                        if v.crossed == 0 and hasattr(v, 'pcb'):
                            v.pcb.priority = max_prio
                            v.speed = speeds.get(v.vehicleClass, 2.0) * 1.2 # Boost speed via inheritance
                else:
                    # Fix 7: Reset priority if no ambulance
                    for v in queue:
                        if hasattr(v, 'pcb'):
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
                ambulance_queue.append((direction_number, self))
                log_event("EMERGENCY", f"Ambulance {self.pcb.vehicle_id} spawned on Signal {direction_number+1}")

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

#Vehicle Movement Synchronization in Vehicle.move():
    def move(self):
        global last_movement_time
        # Keep speed stable (already set in init or signal change)
        pass

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
    features = get_dataset_features(i)
    # Queue length calculation
    direction = directionNumbers[i]
    load = len(vehicles[direction][0]) + \
           len(vehicles[direction][1]) + \
           len(vehicles[direction][2])
    
    # Formula: 50% load + 30% congestion + 20% wait time
    return (0.5 * load) + (0.3 * features["congestion"]) + (0.2 * signals[i].waitTime)

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

    # 0. Fairness Cap (PREVENT STARVATION)
    MAX_WAIT = 60 # Balanced from 50 to 60 for better stability
    for i in range(noOfSignals):
        if i < len(signals) and signals[i].waitTime > MAX_WAIT:
            candidates.append((i, f"[FORCE] Signal {i+1} (Wait > {MAX_WAIT}s)"))
            return candidates # Immediate force
    if SIMULATION_MODE == MODE_MLFQ:
        aging_lanes = []
        for i in range(noOfSignals):
            if i < len(signals) and signals[i].waitTime >= 90:
                aging_lanes.append((i, signals[i].waitTime))
        
        aging_lanes.sort(key=lambda x: x[1], reverse=True)
        for idx, wait in aging_lanes:
            candidates.append((idx, f"[Q0 - Aging] Signal {idx+1} ({wait}s)"))

    # 2. Q1 - OPF Candidates (Fairness by oldest waiting vehicle)
    opf_lane = get_opf_candidate()
    if opf_lane is not None:
        print(f"[DEBUG OPF] Selected Signal {opf_lane+1}")
        candidates.append((opf_lane, f"[Q1 - OPF] Signal {opf_lane+1} (Oldest Process)"))
    else:
        if total_vehicles_waiting > 0:
            print("[WARNING] OPF failed despite vehicles present")
        else:
            print("[DEBUG OPF] No valid candidate")

    # 3. Q2 - MVF Candidates (Throughput Priority)
    if SIMULATION_MODE >= MODE_MVF:
        v_counts = []
        for i in range(noOfSignals):
            if i < len(signals):
                v_counts.append((i, signals[i].active_count))
        v_counts.sort(key=lambda x: x[1], reverse=True)
        for idx, count in v_counts:
            if count > 10:
                candidates.append((idx, f"[Q2 - MVF] Signal {idx+1} ({count} vehicles)"))

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
        # Fix 2: Limit Banker Denial Spam (Retry logic)
        max_attempts = 3
        attempts = 0
        is_granted = False
        while attempts < max_attempts:
            if banker_resource_request(idx):
                is_granted = True
                break
            attempts += 1
            
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

            if "Q0" in source: current_algorithm = "MLFQ (Aging)"
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
    global currentGreen, currentYellow, nextGreen, saved_context, last_interrupt_time, is_in_yellow_phase, is_emergency_active, emergency_chain_count, last_movement_time
    while True:
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
            log_event("CONTEXT", f"Resuming Signal {targetSignal + 1} ({task['remaining']}s)")
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

def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 5)
        if vehicle_type == 5:  # Ambulance
            if len(ambulance_queue) > 2:
                continue # block excessive ambulances
            lane_number = 1  # Ambulances only in lane 1
            if random.random() > 0.75:  # 25% chance for ambulance
                will_turn = 0
                direction_number = random.choice([0, 1, 2, 3])  # Random direction
                Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
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
    global timeElapsed, simTime, lane_load_data, lane_wait_times
    while True:
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
                "banker": banker_stats
            }
            
            mode_name = ["rr", "mvf", "mlfq"][SIMULATION_MODE]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            os.makedirs("results", exist_ok=True)
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

    screenWidth = 1400
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

        screen.blit(background, (0, 0))
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

        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, (1100, 50))
        
        # --- Kernel HUD Overlay (Phase 4 Cleanup) ---
        hud_font = pygame.font.Font(None, 28)
        mode_text = ["Round-Robin", "MVF (Density Only)", "Full OS MLFQ"][SIMULATION_MODE]
        mode_label = hud_font.render(f"KERNEL MODE: {mode_text}", True, (255, 255, 0), black)
        screen.blit(mode_label, (50, 50))
        
        # Display Current Algorithm
        algo_label = hud_font.render(f"ALGORITHM: {current_algorithm}", True, (0, 255, 255), black)
        screen.blit(algo_label, (50, 80))
        
        # Display Current State
        q_label_text = "STATE: IDLE / RED"
        if is_in_yellow_phase:
            q_label_text = f"STATE: CLEARING YELLOW (Signal {currentGreen+1})"
        elif saved_context:
            q_label_text = "STATE: CONTEXT RESTORE"
        elif any(v.pcb.vehicle_type == 'ambulance' and v.crossed == 0 for v in simulation):
            q_label_text = f"STATE: EMERGENCY CONVOY (Signal {currentGreen+1}: INF)"
        elif signals[currentGreen].green > 0:
            q_label_text = f"STATE: GREEN (Time-Based: {signals[currentGreen].green}s)"
        
        state_label = hud_font.render(q_label_text, True, (255, 255, 255), black)
        screen.blit(state_label, (50, 110))
        
        # Display Metrics in HUD
        cleared_text = hud_font.render(f"Vehicles Cleared: {metrics['total_vehicles_cleared']}", True, (200, 200, 200), black)
        screen.blit(cleared_text, (50, 140))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()

if __name__ == "__main__":
    Main()
