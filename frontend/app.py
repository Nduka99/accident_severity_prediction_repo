import streamlit as st
import requests
import json
import os
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="US Accident Severity Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment Variables
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Render 'host' property returns domain only, so we must ensure schema
if not BACKEND_URL.startswith("http"):
    BACKEND_URL = f"https://{BACKEND_URL}"

# --- Mappings (Frontend Label -> Backend Keyword) ---
DESCRIPTION_MAP = {
    "Queueing Traffic": "Queueing traffic detected",
    "Heavy Congestion": "Heavy congestion reported",
    "Lane Blocked": "Lane blocked due to incident",
    "Ramp Incident": "Incident on ramp",
    "Accident Reported": "Accident reported",
    "Hazard on Road": "Hazard on road",
    "Caution Alert": "Caution alert issued",
    "Fire/Smoke": "Fire or smoke detected"
}

WEATHER_OPTIONS = [
    "Clear",
    "Cloudy",
    "Rain",
    "Snow/Ice",
    "Fog/Obscured",
    "Storm"
]

# --- UI Layout ---

# Sidebar
# Sidebar (Removed by request)

# Main Column
st.title("🚦 US Accident Severity Predictor")
st.markdown("""
This application predicts the severity of a traffic accident based on environmental and road conditions.
**Severity 1**: Non-Severe (Minor) | **Severity 0**: Severe (Major)
""")
st.markdown("---")

# Form
with st.form("prediction_form"):
    st.header("1. Accident Context")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Date", value=datetime.now())
        start_time = st.time_input("Time", value=datetime.now().time())
    
    with col2:
        desc_selections = st.multiselect(
            "Incident Description (Select all that apply)",
            options=list(DESCRIPTION_MAP.keys()),
            default=["Accident Reported"],
            help="Select key terms to simulate sensor text feed."
        )

    st.header("2. Environmental Conditions")
    with st.expander("Weather & Road Conditions", expanded=True):
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            weather_condition = st.selectbox("Weather Condition", options=WEATHER_OPTIONS)
            temp = st.slider("Temperature (°F)", -20.0, 110.0, 70.0, format="%.1f")
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, format="%.1f")
        
        with w_col2:
            pressure = st.number_input("Pressure (in)", 20.0, 35.0, 29.9, step=0.1)
            visibility = st.slider("Visibility (miles)", 0.0, 20.0, 10.0, step=0.5)
            wind_speed = st.slider("Wind Speed (mph)", 0.0, 100.0, 10.0, step=1.0)
            
    st.header("3. Infrastructure")
    with st.expander("Road Features"):
        st.write("Select features present at the accident site:")
        poi_cols = st.columns(4)
        
        # POI Variables
        traffic_signal = poi_cols[0].checkbox("Traffic Signal")
        junction = poi_cols[1].checkbox("Junction")
        crossing = poi_cols[2].checkbox("Crossing")
        stop = poi_cols[3].checkbox("Stop Sign")
        
        amenity = poi_cols[0].checkbox("Amenity")
        railway = poi_cols[1].checkbox("Railway")
        station = poi_cols[2].checkbox("Station")
        bump = poi_cols[3].checkbox("Bump")
        
    submit_btn = st.form_submit_button("Predict Severity", type="primary", use_container_width=True)

# --- Logic & API Call ---
if submit_btn:
    # 1. Construct Payload
    # Combine date and time
    combined_dt = datetime.combine(start_date, start_time)
    
    # Concatenate Description keywords
    if not desc_selections:
        final_desc = "Accident" # Fallback
    else:
        # Map selections to backend keywords and join
        final_desc = ". ".join([DESCRIPTION_MAP[s] for s in desc_selections])
        
    payload = {
        "Start_Time": combined_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Description": final_desc,
        "Street": "Simulated St", # Placeholder, not used in model logic but required by schema? Schema says Optional.
        "Weather_Condition": weather_condition,
        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Wind_Direction": "Calm", # Defaulting for simplicity
        "Precipitation(in)": 0.0, # Defaulting
        # POIs
        "Traffic_Signal": traffic_signal,
        "Junction": junction,
        "Crossing": crossing,
        "Stop": stop,
        "Amenity": amenity,
        "Railway": railway,
        "Station": station,
        "Bump": bump,
        # Default others to False
        "Give_Way": False,
        "No_Exit": False,
        "Roundabout": False,
        "Traffic_Calming": False,
        "Turning_Loop": False
    }
    
    # Debug Payload (Optional, can comment out for prod)
    # with st.expander("Debug Payload"):
    #     st.json(payload)
        
    # 2. Call API
    # 2.5 Security: Retrieve Backend "Backstage Pass" to authorize request
    API_SECRET = os.getenv("API_SECRET", "dev-secret")
    headers = {"X-Service-Token": API_SECRET}

    with st.spinner("Analyzing accident data..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict", 
                json=payload,
                headers=headers # <--- Send the password!
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result['severity_probability']
                label = result['prediction_label']
                time_ms = result['processing_time_ms']
                
                st.markdown("### Prediction Results")
                
                # Visuals
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    if label == "Severe":
                        st.error(f"## {label}")
                    else:
                        st.success(f"## {label}")
                
                with res_col2:
                    st.metric("Severity Probability", f"{prob:.2%}", delta=f"{time_ms} ms inference")
                    st.progress(prob)
                    
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not connect to the backend. Is the prediction engine running?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
