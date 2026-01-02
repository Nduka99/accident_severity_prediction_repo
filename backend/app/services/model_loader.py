import joblib
import pandas as pd
from ..config import settings
import os

class ModelLoader:
    _model = None
    _scaler = None

    @classmethod
    def load_models(cls):
        """
        Loads the Model and Scaler from disk into memory.
        This should be called ONCE at application startup.
        """
        if cls._model is None or cls._scaler is None:
            print(f"Loading Model from: {settings.MODEL_PATH}")
            if not os.path.exists(settings.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}")
                
            print(f"Loading Scaler from: {settings.SCALER_PATH}")
            if not os.path.exists(settings.SCALER_PATH):
                raise FileNotFoundError(f"Scaler file not found at {settings.SCALER_PATH}")

            cls._model = joblib.load(settings.MODEL_PATH)
            cls._scaler = joblib.load(settings.SCALER_PATH)
            print("Artifacts loaded successfully.")
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.load_models()
        return cls._model

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            cls.load_models()
        return cls._scaler
