from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class AccidentInput(BaseModel):
    # --- 1. DateTime (Crucial for Cyclical Features) ---
    Start_Time: datetime = Field(..., description="The start time of the accident in ISO format (e.g., 2023-01-01 12:00:00)")
    
    # --- 2. Text (Crucial for Regex Features) ---
    Description: str = Field(..., description="Natural language description of the accident (e.g. 'Queueing traffic due to accident')")
    
    # --- 3. Location / Infrastructure ---
    # Used for Road_Label logic (Highway vs Local)
    Street: Optional[str] = Field(None, description="Street name (used to derive Is_Highway)")
    
    # --- 4. Weather & Environmental (Core Features) ---
    # Note: We use specific names matching training data
    # Weather_Condition is CRITICAL for simplification
    Weather_Condition: str = Field(..., description="Raw weather condition (e.g. 'Light Rain', 'Scattered Clouds')")
    
    # Numerical Weather features
    Temperature_F: float = Field(..., alias="Temperature(F)", description="Temperature in Fahrenheit")
    Humidity_Percent: float = Field(..., alias="Humidity(%)", description="Humidity percentage (0-100)")
    Pressure_in: float = Field(..., alias="Pressure(in)", description="Atmospheric Pressure in inches")
    Visibility_mi: float = Field(..., alias="Visibility(mi)", description="Visibility in miles")
    Wind_Speed_mph: float = Field(..., alias="Wind_Speed(mph)", description="Wind Speed in mph")
    Precipitation_in: float = Field(0.0, alias="Precipitation(in)", description="Precipitation in inches (default 0 if missing)")
    Wind_Direction: str = Field("Calm", description="Wind Direction (e.g. 'WSW', 'N', 'Calm')")

    # --- 5. Points of Interest (Boolean Features) ---
    # These are 0/1 in model, but we accept bool for API friendliness and convert later
    Amenity: bool = False
    Bump: bool = False # Note: Dropped in scaler, but we can accept it to be safe or ignore it.
    Crossing: bool = False
    Give_Way: bool = False
    Junction: bool = False
    No_Exit: bool = False
    Railway: bool = False
    Roundabout: bool = False
    Station: bool = False
    Stop: bool = False
    Traffic_Calming: bool = False
    Traffic_Signal: bool = False
    Turning_Loop: bool = False

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Start_Time": "2023-03-15 08:30:00",
                "Description": "Accident on I-95 North. Queueing traffic.",
                "Street": "I-95 N",
                "Weather_Condition": "Light Rain",
                "Temperature(F)": 45.5,
                "Humidity(%)": 82.0,
                "Pressure(in)": 29.9,
                "Visibility(mi)": 10.0,
                "Wind_Speed(mph)": 12.5,
                "Wind_Direction": "NW",
                "Precipitation(in)": 0.02,
                "Traffic_Signal": True,
                "Junction": False
            }
        }

class PredictionOutput(BaseModel):
    severity_probability: float = Field(..., description="Probability of the accident being Severe (Class 1)")
    prediction_label: str = Field(..., description="Text label: 'Severe' or 'Minor'")
    processing_time_ms: float = Field(..., description="Time taken to process request")
