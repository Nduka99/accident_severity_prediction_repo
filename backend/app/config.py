import os
from pathlib import Path

class Settings:
    # Base Directory: backend/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # Artifacts Directory
    ARTIFACTS_DIR = BASE_DIR / "backend" / "model_artifacts"
    
    # Model Paths
    MODEL_PATH = ARTIFACTS_DIR / "lgbm_tuned_model.pkl"
    SCALER_PATH = ARTIFACTS_DIR / "robust_scaler.pkl"
    
    # App Settings
    APP_NAME = "US Accident Severity Prediction API"
    VERSION = "1.0.0"

settings = Settings()
