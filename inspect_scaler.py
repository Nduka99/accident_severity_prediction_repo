import joblib
import pandas as pd
from backend.app.config import settings
import sys

# Add project root to path so config can be imported
sys.path.append(r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model")

try:
    scaler = joblib.load(settings.SCALER_PATH)
    print(f"Scaler loaded from: {settings.SCALER_PATH}")
    print(f"Expected Feature Count: {scaler.n_features_in_}")
    
    if hasattr(scaler, "feature_names_in_"):
        print("Feature Names (Order is Critical):")
        print(list(scaler.feature_names_in_))
    else:
        print("WARNING: 'feature_names_in_' not found on scaler. Order is unknown!")

except Exception as e:
    print(f"Error: {e}")
