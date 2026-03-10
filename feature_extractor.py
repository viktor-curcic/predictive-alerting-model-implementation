import numpy as np
import pandas as pd
from typing import Tuple

class FeatureExtractor:
    def __init__(self, window_size: int = 60, horizon: int = 10, stride: int = 1):
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        
    def extract_features(self, window: pd.DataFrame) -> np.ndarray:
        features = []
        
        for col in window.columns:
            values = window[col].values
            
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.percentile(values, 95),
                values[-1],  
                values[-1] - values[0],  
            ])
            
            if len(values) > 5:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                features.append(slope)
                
                recent_slope = np.polyfit(x[-10:], values[-10:], 1)[0]
                features.append(recent_slope)
            else:
                features.extend([0, 0])
            
            diffs = np.diff(values)
            features.append(np.std(diffs))
        
        if 'cpu_utilization' in window.columns and 'error_rate' in window.columns:
            features.append(window['cpu_utilization'].iloc[-1] * window['error_rate'].iloc[-1])
        
        last_time = window.index[-1]
        features.extend([
            last_time.hour / 24.0,
            last_time.dayofweek / 7.0,
        ])
        
        return np.array(features)
    
    def create_windows(self, metrics_df: pd.DataFrame, 
                      incidents_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        X_list = []
        y_list = []
        timestamps = []
        
        n = len(metrics_df) - self.window_size - self.horizon + 1
        
        for i in range(0, n, self.stride):
            window = metrics_df.iloc[i:i+self.window_size]
            features = self.extract_features(window)
            X_list.append(features)
            
            prev_value = incidents_df.iloc[i + self.window_size - 1]['incident']
            future = incidents_df.iloc[i+self.window_size:i+self.window_size+self.horizon]['incident'].values
            
            incident_starts = 0
            last = prev_value
            for val in future:
                if val == 1 and last == 0:
                    incident_starts = 1
                    break
                last = val
            
            y_list.append(incident_starts)
            timestamps.append(metrics_df.index[i+self.window_size])
        
        return np.array(X_list), np.array(y_list), pd.DatetimeIndex(timestamps)