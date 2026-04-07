import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed. Skipping XGBoost model.")

import joblib
import os

def train():
    print("[INFO] Loading Dataset for Multi-Model Training...")
    try:
        df = pd.read_csv("Banglore_traffic_Dataset.csv")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Features
    features = [
        'Traffic Volume', 'Average Speed', 'Congestion Level', 
        'Pedestrian and Cyclist Count', 'Incident Reports', 
        'Traffic Signal Compliance', 'Road Capacity Utilization', 
        'Travel Time Index'
    ]

    # Target Engineering (High-Responsiveness Formula)
    df['green_time'] = (
        (df['Traffic Volume'] / 1000) +
        (df['Congestion Level'] / 5) +
        (df['Pedestrian and Cyclist Count'] / 50) +
        (df['Road Capacity Utilization'] / 8) +
        (df['Travel Time Index'] * 3) -
        (df['Average Speed'] / 15)
    )
    df['green_time'] = df['green_time'].clip(5, 25)

    X = df[features]
    y = df['green_time']

    # 1. Linear Regression
    print("[TRAIN] Fitting Linear Regression...")
    lr = LinearRegression()
    lr.fit(X, y)
    joblib.dump(lr, "models/lr.pkl")

    # 2. Decision Tree
    print("[TRAIN] Fitting Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X, y)
    joblib.dump(dt, "models/dt.pkl")

    # 3. Random Forest
    print("[TRAIN] Fitting Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "models/rf.pkl")

    # 4. XGBoost
    if HAS_XGB:
        print("[TRAIN] Fitting XGBoost...")
        xgb = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        xgb.fit(X, y)
        joblib.dump(xgb, "models/xgb.pkl")

    joblib.dump(features, "models/feature_names.pkl")
    print("[SUCCESS] All models saved to 'models/' directory.")

if __name__ == "__main__":
    train()
