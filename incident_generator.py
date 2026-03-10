import numpy as np
import pandas as pd
from typing import Tuple

class BalancedIncidentGenerator:
    
    def __init__(self, window_size: int = 60, horizon: int = 10, random_seed: int = 42):
        self.window_size = window_size
        self.horizon = horizon
        np.random.seed(random_seed)
        
    def generate(self, n_days: int = 90, freq: str = '1min') -> Tuple[pd.DataFrame, pd.DataFrame]:
        n_points = n_days * 24 * 60
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq=freq)
        t = np.arange(n_points)
        
        cpu = 40 + 15 * np.sin(2 * np.pi * t / (24*60)) + np.random.standard_t(df=3, size=n_points) * 5
        mem = 60 + 0.005 * t / n_points * 10 + np.random.normal(0, 3, n_points)
        req = 100 + 30 * np.sin(2 * np.pi * t / (7*24*60)) + np.random.poisson(8, n_points)
        error = np.random.exponential(0.3, n_points)
        
        metrics_df = pd.DataFrame({
            'cpu_utilization': np.clip(cpu, 0, 100),
            'memory_usage': np.clip(mem, 30, 95),
            'request_rate': np.maximum(req, 0),
            'error_rate': error
        }, index=timestamps)
        
        incident_mask = np.zeros(n_points, dtype=int)
        
        for _ in range(250):  
            start_idx = np.random.randint(60, n_points - 60)
            
            incident_type = np.random.choice(['detectable', 'hard', 'unpredictable'], 
                                           p=[0.7, 0.2, 0.1])
            
            if incident_type == 'detectable':
                precursor_start = start_idx - 30
                cpu_ramp = np.linspace(0, 20, 30)
                metrics_df.iloc[precursor_start:start_idx, 0] += cpu_ramp
                mem_ramp = np.linspace(0, 10, 30)
                metrics_df.iloc[precursor_start:start_idx, 1] += mem_ramp
                error_spikes = np.random.poisson(2, 30)
                metrics_df.iloc[precursor_start:start_idx, 3] += error_spikes * 0.5
                
            elif incident_type == 'hard':
                precursor_start = start_idx - 20
                cpu_ramp = np.linspace(0, 8, 20)
                metrics_df.iloc[precursor_start:start_idx, 0] += cpu_ramp
                error_spikes = np.random.poisson(1, 20)
                metrics_df.iloc[precursor_start:start_idx, 3] += error_spikes * 0.3
            
            duration = np.random.randint(10, 41)
            incident_mask[start_idx:min(start_idx + duration, n_points)] = 1
        
        for _ in range(50):
            start_idx = np.random.randint(60, n_points - 60)
            precursor_start = start_idx - 30
            cpu_ramp = np.linspace(0, 15, 30)
            metrics_df.iloc[precursor_start:start_idx, 0] += cpu_ramp * 0.7
        
        metrics_df['cpu_utilization'] = np.clip(metrics_df['cpu_utilization'], 0, 100)
        metrics_df['memory_usage'] = np.clip(metrics_df['memory_usage'], 30, 95)
        metrics_df['error_rate'] = np.maximum(metrics_df['error_rate'], 0)
        
        incidents_df = pd.DataFrame({'incident': incident_mask}, index=timestamps)
        
        return metrics_df, incidents_df