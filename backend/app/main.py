from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
import numpy as np

# Internal Imports
from .schemas import AccidentInput, PredictionOutput
from .config import settings
from .services.model_loader import ModelLoader
from .services.feature_engineering import feature_engine

# 1. Initialize App
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="API for predicting US Accident Severity (LightGBM)"
)

# 2. CORS (Allow Streamlit Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Startup Event (The "Warmup")
@app.on_event("startup")
async def startup_event():
    """
    Load artifacts into memory when the server starts.
    """
    try:
        ModelLoader.load_models()
        print("System ready.")
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: {e}")
        # In production, you might want to force exit here if models fail
        # raise e

# 4. Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ModelLoader._model is not None}

# 5. Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict_severity(input_data: AccidentInput):
    """
    Main inference endpoint.
    1. Validates input (Pydantic)
    2. Transforms features (FeatureEngineer)
    3. Predicts probability (LightGBM)
    """
    start_time = time.time()
    
    try:
        # A. Get Model
        model = ModelLoader.get_model()
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # B. Feature Engineering
        # Transforms Pydantic object -> Numpy Array (1, 54)
        features = feature_engine.transform(input_data)
        
        # C. Inference
        # predict_proba returns [[prob_0, prob_1]] -> We want prob_1 (Severe)
        probs = model.predict_proba(features)
        severe_prob = float(probs[0][1])
        
        # D. Logic for Label
        # Using 0.5 as threshold, but this can be tuned via 'settings' later
        label = "Severe" if severe_prob >= 0.5 else "Minor"
        
        # E. Response
        processing_time = (time.time() - start_time) * 1000 # ms
        
        return PredictionOutput(
            severity_probability=severe_prob,
            prediction_label=label,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Local Dev Run
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
