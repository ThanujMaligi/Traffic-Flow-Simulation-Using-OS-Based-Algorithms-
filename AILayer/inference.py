import joblib
import numpy as np
import os

class MultiModelInference:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultiModelInference, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.feature_names = None
            cls._instance.load_all_models()
        return cls._instance

    def load_all_models(self):
        model_dir = "models"
        model_files = {
            "lr": "lr.pkl",
            "dt": "dt.pkl",
            "rf": "rf.pkl",
            "xgb": "xgb.pkl"
        }
        
        feature_path = os.path.join(model_dir, "feature_names.pkl")
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)
        
        for m_type, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                self.models[m_type] = joblib.load(path)
                print(f"[SUCCESS] Loaded model type: {m_type}")
            else:
                print(f"[WARNING] Model file {filename} not found.")

    def predict_green_time(self, features_dict, model_type="rf"):
        model = self.models.get(model_type)
        if model is None:
            # Fallback to RF if available, else LR, else default 10
            model = self.models.get("rf") or self.models.get("lr")
            if model is None: return 10
            
        try:
            # Feature mapping (Internal -> Dataset)
            key_map = {
                'Traffic Volume': 'volume',
                'Average Speed': 'speed',
                'Congestion Level': 'congestion',
                'Pedestrian and Cyclist Count': 'pedestrians',
                'Incident Reports': 'incidents',
                'Traffic Signal Compliance': 'compliance',
                'Road Capacity Utilization': 'capacity',
                'Travel Time Index': 'tti'
            }
            
            feature_values = []
            for name in self.feature_names:
                internal_key = key_map.get(name, name)
                feature_values.append(features_dict.get(internal_key, 0))
            
            X = np.array([feature_values])
            
            # Suppress feature name warnings for models trained with DataFrames
            prediction = model.predict(X)[0]
            
            return max(5, min(25, float(prediction)))
        except Exception as e:
            print(f"[INFERENCE ERROR] {model_type} failed: {e}")
            return 10

# Singleton
engine = MultiModelInference()

def predict_green_time(features_dict, model_type="rf"):
    return engine.predict_green_time(features_dict, model_type)
