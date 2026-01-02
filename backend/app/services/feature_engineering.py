import pandas as pd
import numpy as np
import re
from datetime import datetime
from ..schemas import AccidentInput
from .model_loader import ModelLoader

class FeatureEngineer:
    # EXACT Column Order from Scaler (Index 0 to 53)
    FEATURE_ORDER = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
        'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Is_Night', 
        'Desc_Queue', 'Desc_Heavy', 'Desc_Blocked', 'Desc_Ramp', 'Desc_Accident', 'Desc_Hazard', 'Desc_Caution', 'Desc_Fire', 
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 'Log_Precipitation(in)', 
        'Weather_Simplified_Cloudy', 'Weather_Simplified_Fog/Obscured', 'Weather_Simplified_Rain', 'Weather_Simplified_Snow/Ice', 'Weather_Simplified_Storm', 
        'Wind_Direction_E', 'Wind_Direction_ENE', 'Wind_Direction_ESE', 'Wind_Direction_N', 
        'Wind_Direction_NE', 'Wind_Direction_NNE', 'Wind_Direction_NNW', 'Wind_Direction_NW', 
        'Wind_Direction_S', 'Wind_Direction_SE', 'Wind_Direction_SSE', 'Wind_Direction_SSW', 
        'Wind_Direction_SW', 'Wind_Direction_VAR', 'Wind_Direction_W', 'Wind_Direction_WNW', 'Wind_Direction_WSW'
    ]

    def transform(self, input_data: AccidentInput) -> np.ndarray:
        """
        Main pipeline: Input Schema -> Scaled Numpy Array (1, 54)
        """
        # 1. Base Data
        data = input_data.dict(by_alias=True)
        
        # 2. Derived Physics Features
        # Wind Chill Formula: 35.74 + 0.6215T - 35.75(V^0.16) + 0.4275T(V^0.16)
        T = data['Temperature(F)']
        V = data['Wind_Speed(mph)']
        # Only valid if T < 50F and V > 3mph, otherwise roughly equal to Temp
        if T < 50 and V > 3:
            data['Wind_Chill(F)'] = 35.74 + (0.6215 * T) - (35.75 * (V ** 0.16)) + (0.4275 * T * (V ** 0.16))
        else:
            data['Wind_Chill(F)'] = T
            
        # 3. Derived Time Features
        dt = input_data.Start_Time
        hour = dt.hour
        month = dt.month
        
        data['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
        data['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
        data['Month_Sin'] = np.sin(2 * np.pi * month / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * month / 12)
        
        # Is_Night Heuristic (6PM to 6AM)
        # Note: Scaler expects 0 or 1
        data['Is_Night'] = 1 if (hour >= 18 or hour < 6) else 0

        # 4. Text Features (Regex)
        desc = input_data.Description.lower()
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
        for key, pattern in keywords.items():
            data[key] = 1 if re.search(pattern, desc) else 0

        # 5. Log Precipitation
        # log1p(0) = 0
        data['Log_Precipitation(in)'] = np.log1p(data['Precipitation(in)'])

        # 6. Categorical One-Hot Encoding (Manual Alignment)
        # A. Weather
        w = input_data.Weather_Condition.lower()
        simple_w = 'Clear' # Default
        if any(x in w for x in ['snow', 'sleet', 'ice', 'freezing', 'wintry', 'hail']): simple_w = 'Snow/Ice'
        elif any(x in w for x in ['thunder', 't-storm', 'tornado', 'squall']): simple_w = 'Storm'
        elif any(x in w for x in ['rain', 'drizzle', 'shower']): simple_w = 'Rain'
        elif any(x in w for x in ['fog', 'mist', 'haze', 'smoke', 'dust', 'sand']): simple_w = 'Fog/Obscured'
        elif any(x in w for x in ['cloudy', 'overcast']): simple_w = 'Cloudy'
        
        # Set all Weather_Simplified_* to 0
        for feat in self.FEATURE_ORDER:
            if 'Weather_Simplified_' in feat:
                data[feat] = 0
        # Set active one to 1 (if not Clear)
        target_col = f"Weather_Simplified_{simple_w}"
        if target_col in self.FEATURE_ORDER: # 'Clear' won't be in list
            data[target_col] = 1

        # B. Wind Direction
        wd = input_data.Wind_Direction.upper()
        # Set all Wind_Direction_* to 0
        for feat in self.FEATURE_ORDER:
            if 'Wind_Direction_' in feat:
                data[feat] = 0
        # Set active one to 1
        target_wd = f"Wind_Direction_{wd}"
        if target_wd in self.FEATURE_ORDER:
            data[target_wd] = 1
        
        # 7. Boolean Casting (POI)
        # Ensure Schema bools become ints
        poi_cols = ['Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
        for p in poi_cols:
            data[p] = 1 if data.get(p, False) else 0

        # 8. Assemble Vector (Strict Order)
        feature_vector = []
        for feature in self.FEATURE_ORDER:
            val = data.get(feature)
            if val is None:
                # Fallback for anything missed (should not happen with strict schema + logic)
                print(f"WARNING: Feature {feature} missing in processing. Defaulting to 0.")
                val = 0
            feature_vector.append(float(val))
            
        # 9. Scale
        X = np.array([feature_vector]) # Shape (1, 54)
        scaler = ModelLoader.get_scaler()
        
        return scaler.transform(X)

feature_engine = FeatureEngineer()
