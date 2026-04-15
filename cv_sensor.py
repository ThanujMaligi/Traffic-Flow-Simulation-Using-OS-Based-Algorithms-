import json
import time
import cv2
import os
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
VIDEO_PATH = "traffic_video.mp4"
JSON_PATH = "cv_traffic_counts.json"
REPO_ID = "Perception365/VehicleNet-Y26s" # The Hugging Face UVH-26 Model Repo
MODEL_FILENAME = "VehicleNet-Y26s.pt" # Actual weights file name in the HF repo

print("[CV SENSOR] Initializing Digital Twin Sensor...")

# 1. Automatically Download the IISc Bengaluru UVH-26 Model from Hugging Face
print(f"[CV SENSOR] Connecting to Hugging Face: {REPO_ID}")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    print(f"[CV SENSOR] Model downloaded/loaded successfully: {model_path}")
except Exception as e:
    print(f"[CV SENSOR] WARNING: Could not download from HF. Using default YOLOv8n.pt as fallback.")
    model_path = "yolov8n.pt"

# Load the Model
model = YOLO(model_path)

# 2. Check if video exists
if not os.path.exists(VIDEO_PATH):
    print(f"\n[ERROR] Video file '{VIDEO_PATH}' not found!")
    print("Please download a short YouTube traffic video and save it in this folder as 'traffic_video.mp4'.")
    print("Waiting for video file...")
    while not os.path.exists(VIDEO_PATH):
        time.sleep(2)
    print("[CV SENSOR] Video detected! Starting processing...\n")

cap = cv2.VideoCapture(VIDEO_PATH)

# Video Dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- SPARSE SENSOR ROI DEFINITIONS ---
# ROI_NORTHBOUND: Covers the right lane (upward-bound traffic)
ROI_NORTHBOUND = np.array([
    [980, 100],
    [1820, 100],
    [1870, 1070],
    [980, 1070]
], dtype=np.int32)

# ROI_SOUTHBOUND: Covers the left lane (downward-bound traffic)
ROI_SOUTHBOUND = np.array([
    [100, 100],
    [940, 100],
    [940, 1070],
    [50, 1070]
], dtype=np.int32)

# YOLO class ID mappings to Pygame-compatible vehicle names
CLASS_MAPPING = {
    2: 'car',
    3: 'bike',
    5: 'bus',
    7: 'truck'
}

seen_ids = {"lane_0": set(), "lane_1": set(), "lane_2": set(), "lane_3": set()}

print("[CV SENSOR] Live Tracking Started. Press 'q' to stop.")

frame_count = 0
# Scenario A list-based buffers
new_arrivals_buffer = {"lane_0": [], "lane_1": [], "lane_2": [], "lane_3": []}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video
        
    frame_count += 1
    
    # Process only every 3rd frame to match real-time processing speed on CPU
    if frame_count % 3 != 0:
        continue
    
    # Run YOLO tracker. 'persist=True' remembers object IDs between frames.
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False, imgsz=320) # tracking cars, bikes, buses, trucks
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() # Center x, Center y, width, height
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            cx, cy, w, h = box
            cx, cy = float(cx), float(cy)
            
            # ROI classification using pointPolygonTest
            lane = None
            if cv2.pointPolygonTest(ROI_NORTHBOUND, (cx, cy), False) >= 0:
                lane = "lane_0"
            elif cv2.pointPolygonTest(ROI_SOUTHBOUND, (cx, cy), False) >= 0:
                lane = "lane_2"
            
            if lane is not None:
                # If we've never seen this vehicle in this lane before, count it!
                if track_id not in seen_ids[lane]:
                    seen_ids[lane].add(track_id)
                    v_type_str = CLASS_MAPPING.get(class_id, 'car')
                    new_arrivals_buffer[lane].append(v_type_str)
                    
    # Visualize ROI Boundaries and Centroids
    cv2.polylines(frame, [ROI_NORTHBOUND], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(frame, [ROI_SOUTHBOUND], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Text overlays for visual telemetry
    cv2.putText(frame, "NORTHBOUND (REAL - Lane 0)", (990, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, "SOUTHBOUND (REAL - Lane 2)", (110, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    annotated_frame = results[0].plot()
    
    # We should overlay the ROI drawings and cumulative metrics on the annotated output
    cv2.polylines(annotated_frame, [ROI_NORTHBOUND], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(annotated_frame, [ROI_SOUTHBOUND], isClosed=True, color=(0, 0, 255), thickness=2)
    
    nb_count = len(seen_ids["lane_0"])
    sb_count = len(seen_ids["lane_2"])
    cv2.putText(annotated_frame, f"Northbound Total: {nb_count}", (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Southbound Total: {sb_count}", (50, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Show the video feed (Optional, you can comment this out to run headlessly)
    cv2.imshow("Bengaluru CV Sensor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # Every 1 second (approx 30 frames), send the new arrivals to Pygame JSON Bridge
    if frame_count % 30 == 0:
        try:
            with open(JSON_PATH, "w") as f:
                json.dump(new_arrivals_buffer, f)
            print(f"[LIVE BRIDGE] Sent new arrivals to Pygame: {new_arrivals_buffer}")
            # Reset buffer for the next second (as list-based)
            new_arrivals_buffer = {"lane_0": [], "lane_1": [], "lane_2": [], "lane_3": []}
        except Exception as e:
            pass # Ignore lock errors

cap.release()
cv2.destroyAllWindows()
print("[CV SENSOR] Process Terminated.")

