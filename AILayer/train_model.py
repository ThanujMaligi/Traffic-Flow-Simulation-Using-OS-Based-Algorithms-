import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train():
    print("[INFO] Loading Dataset: Banglore_traffic_Dataset.csv")
    try:
        df = pd.read_csv("Banglore_traffic_Dataset.csv")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Define Features
    features = [
        'Traffic Volume', 
        'Average Speed', 
        'Congestion Level', 
        'Pedestrian and Cyclist Count', 
        'Incident Reports', 
        'Traffic Signal Compliance', 
        'Road Capacity Utilization', 
        'Travel Time Index'
    ]

    # Target Engineering (Optimized for Wait Time Reduction)
    print("[INFO] Engineering High-Responsiveness Target Labels...")
    df['green_time'] = (
        (df['Traffic Volume'] / 1000) +
        (df['Congestion Level'] / 5) +
        (df['Pedestrian and Cyclist Count'] / 50) +
        (df['Road Capacity Utilization'] / 8) +
        (df['Travel Time Index'] * 3) -
        (df['Average Speed'] / 15)
    )

    # Clamp target between [5, 25]
    df['green_time'] = df['green_time'].clip(5, 25)

    X = df[features]
    y = df['green_time']

    # Initialize and Train Model
    print("[INFO] Training Random Forest Regressor (n=150, depth=10)...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X, y)

    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump(model, "models/traffic_rf_model.pkl")
    joblib.dump(features, "models/feature_names.pkl")
    print("[SUCCESS] Model saved to 'models/traffic_rf_model.pkl'")

if __name__ == "__main__":
    train()
