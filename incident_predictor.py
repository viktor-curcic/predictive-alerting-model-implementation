import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score,
                             confusion_matrix, f1_score)
import xgboost as xgb
from typing import Tuple, Dict, List
from feature_extractor import FeatureExtractor

class IncidentPredictor:
    
    def __init__(self, window_size: int = 60, horizon: int = 10):
        self.window_size = window_size
        self.horizon = horizon
        self.model = None
        self.threshold = None
        self.feature_extractor = FeatureExtractor(window_size, horizon, stride=1)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        return {
            'y_val_proba': y_val_proba,
            'val_ap': average_precision_score(y_val, y_val_proba)
        }
    
    def find_best_threshold(self, y_val: np.ndarray, y_val_proba: np.ndarray,
                           incidents_df: pd.DataFrame, val_timestamps: pd.DatetimeIndex,
                           target_incident_recall: float = 0.7) -> float:
        thresholds = np.linspace(0.05, 0.95, 50)
        
        val_start = val_timestamps.min()
        val_end = val_timestamps.max()
        
        incidents_val = incidents_df.loc[val_start:val_end].copy()
        
        best_threshold = 0.5
        best_false_alarms = float('inf')
        
        for threshold in thresholds:
            raw_alerts = y_val_proba >= threshold
            
            pred_df = pd.DataFrame({
                'probability': y_val_proba,
                'raw_alert': raw_alerts
            }, index=val_timestamps)
            
            pred_df = self.apply_alert_cooldown(pred_df, threshold)
            
            incident_metrics = self.evaluate_incident_level(
                incidents_val, pred_df, threshold, val_start, val_end
            )
            
            if incident_metrics['incident_detection_rate'] >= target_incident_recall:
                if incident_metrics['false_alarms_per_day'] < best_false_alarms:
                    best_false_alarms = incident_metrics['false_alarms_per_day']
                    best_threshold = threshold
        
        if best_false_alarms == float('inf'):
            best_recall = 0
            for threshold in thresholds:
                raw_alerts = y_val_proba >= threshold
                pred_df = pd.DataFrame({'probability': y_val_proba, 'raw_alert': raw_alerts}, 
                                      index=val_timestamps)
                pred_df = self.apply_alert_cooldown(pred_df, threshold)
                incident_metrics = self.evaluate_incident_level(
                    incidents_val, pred_df, threshold, val_start, val_end
                )
                if incident_metrics['incident_detection_rate'] > best_recall:
                    best_recall = incident_metrics['incident_detection_rate']
                    best_threshold = threshold
        
        return best_threshold
    
    def threshold_sensitivity_analysis(self, y_val: np.ndarray, y_val_proba: np.ndarray,
                                      incidents_df: pd.DataFrame, 
                                      val_timestamps: pd.DatetimeIndex,
                                      thresholds: List[float]) -> pd.DataFrame:
        val_start = val_timestamps.min()
        val_end = val_timestamps.max()
        incidents_val = incidents_df.loc[val_start:val_end].copy()
        
        results = []
        
        for threshold in thresholds:
            raw_alerts = y_val_proba >= threshold
            pred_df = pd.DataFrame({
                'probability': y_val_proba,
                'raw_alert': raw_alerts
            }, index=val_timestamps)
            
            pred_df = self.apply_alert_cooldown(pred_df, threshold)
            
            metrics = self.evaluate_incident_level(
                incidents_val, pred_df, threshold, val_start, val_end
            )
            
            results.append({
                'Threshold': f'{threshold:.2f}',
                'Incident Detection': f"{metrics['incident_detection_rate']:.1%}",
                'False Alarms/Day': f"{metrics['false_alarms_per_day']:.1f}",
                'Mean Lead Time': f"{metrics['mean_lead_time']:.1f} min"
            })
        
        return pd.DataFrame(results)
    
    def apply_alert_cooldown(self, predictions_df: pd.DataFrame, 
                            threshold: float, cooldown_minutes: int = 15) -> pd.DataFrame:
        df = predictions_df.copy()
        df['raw_alert'] = df['probability'] >= threshold
        df['alert'] = False
        
        last_alert_time = None
        for idx, row in df.iterrows():
            if row['raw_alert']:
                if last_alert_time is None:
                    df.loc[idx, 'alert'] = True
                    last_alert_time = idx
                else:
                    time_since = (idx - last_alert_time).total_seconds() / 60
                    if time_since >= cooldown_minutes:
                        df.loc[idx, 'alert'] = True
                        last_alert_time = idx
        
        return df
    
    def evaluate_window_level(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        return {
            'precision': tp / (tp + fp + 1e-9),
            'recall': tp / (tp + fn + 1e-9),
            'f1': f1_score(y_test, y_pred),
            'false_positive_rate': fp / (fp + tn + 1e-9),
            'average_precision': average_precision_score(y_test, y_proba),
            'threshold': self.threshold,
            'positive_rate': y_test.mean(),
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
            'y_proba': y_proba,
            'y_pred': y_pred
        }
    
    def get_incident_intervals(self, incidents_df: pd.DataFrame, 
                              start_time: pd.Timestamp, 
                              end_time: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        subset = incidents_df.loc[start_time:end_time]
        
        intervals = []
        in_incident = False
        interval_start = None
        
        for idx, row in subset.iterrows():
            if row['incident'] == 1 and not in_incident:
                interval_start = idx
                in_incident = True
            elif row['incident'] == 0 and in_incident:
                intervals.append((interval_start, idx))
                in_incident = False
        
        if in_incident:
            intervals.append((interval_start, end_time))
        
        return intervals
    
    def evaluate_incident_level(self, incidents_df: pd.DataFrame, 
                               predictions_df: pd.DataFrame,
                               threshold: float,
                               eval_start: pd.Timestamp,
                               eval_end: pd.Timestamp) -> Dict:
        buffer_start = eval_start - pd.Timedelta(minutes=30)
        all_intervals = self.get_incident_intervals(incidents_df, buffer_start, eval_end)
        
        incident_intervals = [
            (start, end) for start, end in all_intervals 
            if eval_start <= start <= eval_end
        ]
        
        if not incident_intervals:
            return {
                'total_incidents': 0,
                'incidents_detected': 0,
                'incident_detection_rate': 0,
                'mean_lead_time': 0,
                'median_lead_time': 0,
                'false_alarms': 0,
                'false_alarms_per_day': 0,
                'lead_times': []
            }
        
        detected = 0
        lead_times = []
        
        for start_time, end_time in incident_intervals:
            window_start = start_time - pd.Timedelta(minutes=self.horizon)
            
            alerts_before = predictions_df[
                (predictions_df.index >= window_start) & 
                (predictions_df.index < start_time) &
                (predictions_df['alert'] == True)
            ]
            
            if len(alerts_before) > 0:
                detected += 1
                first_alert = alerts_before.index[0]
                lead_time = (start_time - first_alert).total_seconds() / 60
                lead_times.append(lead_time)
        
        false_alarms = 0
        for idx, row in predictions_df.iterrows():
            if row['alert']:
                future_window = predictions_df[
                    (predictions_df.index > idx) & 
                    (predictions_df.index <= idx + pd.Timedelta(minutes=self.horizon))
                ]
                
                incident_starts_next = False
                for future_idx in future_window.index:
                    if future_idx in incidents_df.index:
                        prev_idx = incidents_df.index[max(0, incidents_df.index.get_loc(future_idx) - 1)]
                        if (incidents_df.loc[future_idx, 'incident'] == 1 and 
                            incidents_df.loc[prev_idx, 'incident'] == 0):
                            incident_starts_next = True
                            break
                
                if not incident_starts_next:
                    false_alarms += 1
        
        days_in_eval = (eval_end - eval_start).total_seconds() / (24 * 3600)
        
        return {
            'total_incidents': len(incident_intervals),
            'incidents_detected': detected,
            'incident_detection_rate': detected / len(incident_intervals),
            'mean_lead_time': np.mean(lead_times) if lead_times else 0,
            'median_lead_time': np.median(lead_times) if lead_times else 0,
            'false_alarms': false_alarms,
            'false_alarms_per_day': false_alarms / max(days_in_eval, 0.1),
            'lead_times': lead_times
        }
    
    def plot_lead_time_distribution(self, lead_times: List[float]):
        if not lead_times:
            print("   No lead times to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(lead_times, bins=15, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(lead_times), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(lead_times):.1f} min')
        plt.axvline(np.median(lead_times), color='g', linestyle='--',
                   label=f'Median: {np.median(lead_times):.1f} min')
        plt.xlabel('Lead Time (minutes)')
        plt.ylabel('Frequency')
        plt.title(f'Alert Lead Time Distribution (Target H={self.horizon} min)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lead_time_distribution.png')
        plt.show()
