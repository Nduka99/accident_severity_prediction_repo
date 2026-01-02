import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import re
import os

# Paths
INPUT_CSV = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\us_accidents_2023_cleaned.csv"
OUTPUT_SCALER = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\backend\model_artifacts\robust_scaler.pkl"

def create_scaler():
    print("Loading data (100k sample)...")
    try:
        # Load enough rows to ensure we capture all categories for OHE if possible
        df = pd.read_csv(INPUT_CSV, nrows=200000)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print("Pre-processing (Strict Replication of Notebook)...")
    
    # --- PHASE 1: Feature Engineering (matches FeatureEngineering class) ---
    
    # 1. Text Features (Regex)
    if 'Description' in df.columns:
        df['desc_clean'] = df['Description'].str.lower().fillna('')
        keywords = {
            'Desc_Queue': r'\b(queue|backups?|slow|stationary|stop|waiting|delays?)\b',
            'Desc_Heavy': r'\b(heavy|congestion|gridlock|bumper)\b',
            'Desc_Blocked': r'\b(block|close|lane|closed|shut|down)\b',
            'Desc_Ramp': r'\b(ramp|exit|entry|interchange)\b',
            'Desc_Accident': r'\b(accident|crash|collision|incident)\b',
            'Desc_Hazard': r'\b(hazard|debris|object|spill|obstacle|animal)\b',
            'Desc_Caution': r'\b(caution|care|alert|warning)\b',
            'Desc_Fire': r'\b(fire|smoke|flame|burn)\b'
        }
        for col, pattern in keywords.items():
            df[col] = df['desc_clean'].str.contains(pattern, regex=True).astype(int)
        df = df.drop(columns=['Description', 'desc_clean'])

    # 2. Weather Simplification
    def simplify_weather(w):
        if pd.isna(w): return 'Clear'
        w = str(w).lower()
        if any(x in w for x in ['snow', 'sleet', 'ice', 'freezing', 'wintry', 'hail']): return 'Snow/Ice'
        if any(x in w for x in ['thunder', 't-storm', 'tornado', 'squall']): return 'Storm'
        if any(x in w for x in ['rain', 'drizzle', 'shower']): return 'Rain'
        if any(x in w for x in ['fog', 'mist', 'haze', 'smoke', 'dust', 'sand']): return 'Fog/Obscured'
        if any(x in w for x in ['cloudy', 'overcast']): return 'Cloudy'
        return 'Clear'

    if 'Weather_Condition' in df.columns:
        df['Weather_Simplified'] = df['Weather_Condition'].apply(simplify_weather)
        # Drop original Weather_Condition as notebook does
        df = df.drop(columns=['Weather_Condition'])

    # 3. Cyclical Time
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        hour = df['Start_Time'].dt.hour
        month = df['Start_Time'].dt.month
        
        df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
        df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * month / 12)
    
    # 4. Log Precipitation
    if 'Precipitation(in)' in df.columns:
        df['Log_Precipitation(in)'] = np.log1p(df['Precipitation(in)'].fillna(0))
        df = df.drop(columns=['Precipitation(in)'])
        
    # 5. Boolean Standardization (POI)
    # Note: 'Bump' is dropped due to multicollinearity (matches notebook logic)
    poi_cols = ['Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 
                'Traffic_Signal', 'Turning_Loop']
    if 'Bump' in df.columns:
        df = df.drop(columns=['Bump'])
        
    for p in poi_cols:
        if p in df.columns:
            df[p] = df[p].astype(int)

    # --- PHASE 2: DROPS (matches Notebook df_trans -> df_trans2 transition) ---
    cols_to_drop = ['City', 'County', 'State', 'Start_Time', 'End_Time', 
                    'Start_Lat', 'Start_Lng', 'Street', 'Hour', 'Month',
                    'End_Lat', 'End_Lng', 
                    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
                    'Timezone', 'Airport_Code', 'Country', 'Weather_Timestamp',
                    'ID', 'Source', 'Zipcode', 'Number']
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # --- PHASE 3: ONE-HOT ENCODING (matches pd.get_dummies call) ---
    # Notebook: pd.get_dummies(df_trans2, columns=['Weather_Simplified', 'Wind_Direction'], drop_first=True, dtype=int)
    
    # Ensure columns exist before encoding
    encode_cols = []
    if 'Weather_Simplified' in df.columns: encode_cols.append('Weather_Simplified')
    if 'Wind_Direction' in df.columns: encode_cols.append('Wind_Direction')
    
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)
        
    # --- PHASE 4: FINAL MODEL DATA PREP (matches ModelTraining class) ---
    # Drop Target Leaks/Labels
    target_drops = ['Severity', 'Binary_Severity', 'Duration_Group', 'Duration_Minutes', 'Distance(mi)']
    df = df.drop(columns=[c for c in target_drops if c in df.columns], errors='ignore')

    # Select ONLY numeric types (Final Filter)
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Handle Final NaNs (RobustScaler needs clean data)
    # The notebook implies data was clean, but for artifact generation we safeguard
    df_numeric = df_numeric.fillna(0)
    
    print(f"Fitting Scaler on {df_numeric.shape[1]} columns...")
    print("Columns:", df_numeric.columns.tolist())
    
    scaler = RobustScaler()
    scaler.fit(df_numeric)
    
    print(f"Saving scaler to {OUTPUT_SCALER}")
    joblib.dump(scaler, OUTPUT_SCALER)
    print("Done.")

if __name__ == "__main__":
    create_scaler()
