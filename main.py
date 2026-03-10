import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, average_precision_score)
from baseline import RuleBasedBaseline
from incident_generator import BalancedIncidentGenerator
from incident_predictor import IncidentPredictor
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

def evaluate_rule_baseline(metrics_df: pd.DataFrame, incidents_df: pd.DataFrame,
                          window_size: int, horizon: int) -> Dict:
    rule_model = RuleBasedBaseline(horizon)
    
    y_true = []
    y_proba = []
    
    n = len(metrics_df) - window_size - horizon + 1
    
    for i in range(0, n, 5):  
        window = metrics_df.iloc[i:i+window_size]
        
        score = rule_model.predict(window)
        y_proba.append(score)
        
        prev_value = incidents_df.iloc[i + window_size - 1]['incident']
        future = incidents_df.iloc[i+window_size:i+window_size+horizon]['incident'].values
        
        incident_starts = 0
        last = prev_value
        for val in future:
            if val == 1 and last == 0:
                incident_starts = 1
                break
            last = val
        
        y_true.append(incident_starts)
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred = (y_proba >= best_threshold).astype(int)
    
    return {
        'name': 'Rule-Based Baseline',
        'average_precision': average_precision_score(y_true, y_proba),
        'best_threshold': best_threshold,
        'recall_at_best': recall[np.argmax(f1_scores)],
        'precision_at_best': precision[np.argmax(f1_scores)]
    }


def main():
    
    print("\n1. GENERATING SYNTHETIC DATA")
    
    generator = BalancedIncidentGenerator(window_size=60, horizon=10)
    metrics_df, incidents_df = generator.generate(n_days=90)
    
    print(f"   Generated {len(metrics_df):,} time steps")
    print(f"   Raw incident rate: {incidents_df['incident'].mean():.2%}")
    
    print("\n2. FEATURE ENGINEERING")

    predictor = IncidentPredictor(window_size=60, horizon=10)
    X, y, timestamps = predictor.feature_extractor.create_windows(metrics_df, incidents_df)
    
    print(f"   Window size (W): 60 minutes")
    print(f"   Prediction horizon (H): 10 minutes")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Total samples: {len(X):,}")
    print(f"   Positive rate (incident starts): {y.mean():.2%}")
    
    print("\n3. TIME-BASED SPLIT")

    n_train = int(len(X) * 0.6)
    n_val = int(len(X) * 0.2)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    val_timestamps = timestamps[n_train:n_train+n_val]
    test_timestamps = timestamps[n_train+n_val:]
    
    print(f"   Training:   {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test:       {len(X_test):,} samples")
    
    print("\n4. BASELINE 1: RULE-BASED")
    
    rule_metrics = evaluate_rule_baseline(metrics_df, incidents_df, 60, 10)
    print(f"   Average Precision: {rule_metrics['average_precision']:.4f}")
    print(f"   Best threshold: {rule_metrics['best_threshold']:.2f}")
    print(f"   Recall at best F1: {rule_metrics['recall_at_best']:.2%}")
    print(f"   Precision at best F1: {rule_metrics['precision_at_best']:.2%}")
    
    print("\n5. BASELINE 2: LOGISTIC REGRESSION")
    
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_val_proba_lr = lr.predict_proba(X_val_scaled)[:, 1]
    lr_ap = average_precision_score(y_val, y_val_proba_lr)
    print(f"   Validation Average Precision: {lr_ap:.4f}")
    
    print("\n6. MAIN MODEL: XGBOOST")
    
    val_results = predictor.train(X_train, y_train, X_val, y_val)
    print(f"   Validation Average Precision: {val_results['val_ap']:.4f}")
    
    print("\n7. THRESHOLD SELECTION")
    
    val_start = val_timestamps.min()
    val_end = val_timestamps.max()
    
    temp_pred_df = pd.DataFrame({
        'probability': val_results['y_val_proba'],
        'raw_alert': val_results['y_val_proba'] >= 0.5
    }, index=val_timestamps)
    temp_pred_df = predictor.apply_alert_cooldown(temp_pred_df, 0.5)
    
    val_incident_metrics = predictor.evaluate_incident_level(
        incidents_df, temp_pred_df, 0.5, val_start, val_end
    )
    
    print(f"\n   Validation incident stats (at threshold 0.5):")
    print(f"   - Incidents in validation: {val_incident_metrics['total_incidents']}")
    print(f"   - Detection rate: {val_incident_metrics['incident_detection_rate']:.1%}")
    print(f"   - False alarms/day: {val_incident_metrics['false_alarms_per_day']:.1f}")
    print(f"   - Mean lead time: {val_incident_metrics['mean_lead_time']:.1f} min")
    
    best_threshold = predictor.find_best_threshold(
        y_val, val_results['y_val_proba'], 
        incidents_df, val_timestamps,
        target_incident_recall=0.7
    )
    predictor.threshold = best_threshold
    print(f"\n   Selected threshold: {best_threshold:.4f}")
    
    print("\n7b. THRESHOLD SENSITIVITY ANALYSIS")
    
    thresholds_to_test = [0.25, 0.35, 0.45, 0.55, 0.65]
    sensitivity_df = predictor.threshold_sensitivity_analysis(
        y_val, val_results['y_val_proba'],
        incidents_df, val_timestamps,
        thresholds_to_test
    )
    
    print("\n   " + sensitivity_df.to_string(index=False))
    
    print("\n8. WINDOW-LEVEL TEST EVALUATION")
    
    window_metrics = predictor.evaluate_window_level(X_test, y_test)
    y_proba = window_metrics['y_proba']
    
    print(f"\n   Test Results (threshold = {window_metrics['threshold']:.4f}):")
    print(f"   - Recall:              {window_metrics['recall']:.2%}")
    print(f"   - Precision:           {window_metrics['precision']:.2%}")
    print(f"   - F1 Score:            {window_metrics['f1']:.3f}")
    print(f"   - Average Precision:   {window_metrics['average_precision']:.4f}")
    
    print("\n9. INCIDENT-LEVEL TEST EVALUATION")
    
    predictions_df = pd.DataFrame({
        'probability': y_proba,
        'raw_alert': (y_proba >= best_threshold)
    }, index=test_timestamps)
    
    predictions_df = predictor.apply_alert_cooldown(predictions_df, best_threshold)
    
    test_start = test_timestamps.min()
    test_end = test_timestamps.max()
    
    incident_metrics = predictor.evaluate_incident_level(
        incidents_df, predictions_df, best_threshold, test_start, test_end
    )
    
    print(f"\n   Test Period: {test_start.date()} to {test_end.date()}")
    print(f"   Incidents starting in test period: {incident_metrics['total_incidents']}")
    print(f"   Detected before start: {incident_metrics['incidents_detected']} "
          f"({incident_metrics['incident_detection_rate']:.1%})")
    print(f"   Mean lead time: {incident_metrics['mean_lead_time']:.1f} minutes")
    print(f"   Median lead time: {incident_metrics['median_lead_time']:.1f} minutes")
    print(f"   False alarms (with cooldown): {incident_metrics['false_alarms']}")
    print(f"   False alarms per day: {incident_metrics['false_alarms_per_day']:.1f}")
    
    print("\n10. LEAD TIME DISTRIBUTION")
    if incident_metrics['lead_times']:
        predictor.plot_lead_time_distribution(incident_metrics['lead_times'])
    else:
        print("   No incidents detected, cannot plot lead times")
    
    print("\n11. MODEL COMPARISON")
    print(f"\n   {'Model':<25} {'AP':<10} {'Recall':<10} {'Precision':<10}")
    print(f"   {'-'*55}")
    print(f"   {'Rule-Based':<25} {rule_metrics['average_precision']:.4f}     "
          f"{rule_metrics['recall_at_best']:.1%}     {rule_metrics['precision_at_best']:.1%}")
    print(f"   {'Logistic Regression':<25} {lr_ap:.4f}     {'N/A':<10} {'N/A':<10}")
    print(f"   {'XGBoost':<25} {window_metrics['average_precision']:.4f}     "
          f"{window_metrics['recall']:.1%}     {window_metrics['precision']:.1%}")
    
    return predictor, window_metrics, incident_metrics, rule_metrics, lr_ap, sensitivity_df


if __name__ == "__main__":
    predictor, window_metrics, incident_metrics, rule_metrics, lr_ap, sensitivity_df = main()