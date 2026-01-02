import requests
import time
import sys
import subprocess
import os

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url)
            return True
        except requests.ConnectionError:
            time.sleep(1)
    return False

def test_prediction():
    url = "http://localhost:8000/predict"
    
    # "Severe" Scenario: Night, Storm, High Wind, blockage
    payload = {
        "Start_Time": "2023-12-25 23:30:00", # Night
        "Description": "Road blocked due to heavy accident. Queueing traffic.", # Keywords: Blocked, Accident, Queue
        "Street": "I-95 N",
        "Weather_Condition": "Heavy Thunderstorms", # Simp: Storm
        "Temperature(F)": 30.0, # Cold -> Ice potential?
        "Humidity(%)": 90.0,
        "Pressure(in)": 29.0, # Low pressure = Storm
        "Visibility(mi)": 0.5, # Poor visibility
        "Wind_Speed(mph)": 35.0, # High wind
        "Precipitation(in)": 0.5,
        "Wind_Direction": "NW",
        "Traffic_Signal": False,
        "Junction": False
    }
    
    print(f"Sending payload: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\n--- API RESPONSE ---")
        print(f"Status Code: {response.status_code}")
        print(f"Prediction Label: {result['prediction_label']}")
        print(f"Probability: {result['severity_probability']:.4f}")
        print(f"Processing Time: {result['processing_time_ms']} ms")
        
        # Simple assertion
        if "severity_probability" in result:
            print("\n[SUCCESS] API returned a valid prediction.")
        else:
            print("\n[FAILURE] Response missing key fields.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(e.response.text)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure server is up
    health_url = "http://localhost:8000/health"
    print("Waiting for server...")
    if wait_for_server(health_url):
        print("Server is ready.")
        test_prediction()
    else:
        print("Server failed to start.")
        sys.exit(1)
