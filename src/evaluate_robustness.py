import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.stress_tester import inject_gaussian_noise
from pathlib import Path

def run_leak_proof_robustness_test(data_path: str, model_artifact_path: str):
    print("Loading production model artifact and master features...")
    payload = joblib.load(model_artifact_path)
    model = payload["model"]
    baseline_threshold = payload["optimal_threshold"]
    feature_names = payload["feature_names"]
    
    #* raw dataframe
    df = pd.read_csv(data_path)
    
    #! target colm
    target_col = "Machine failure" if "Machine failure" in df.columns else "Machine_failure"
    y = df[target_col]
    
    #todo Check for feature alignments and map naming variations
    available_features = [col for col in feature_names if col in df.columns]
    
    if len(available_features) < len(feature_names):
        print(f"Naming variance detected. Aligning model feature names with dataframe headers...")
        mapping = {f: f.replace('_', ' ').replace(' K', ' [K]').replace(' rpm', ' [rpm]').replace(' Nm', ' [Nm]').replace(' min', ' [min]') for f in feature_names}
        X_df_columns = []
        for f in feature_names:
            if f in df.columns:
                X_df_columns.append(f)
            elif mapping.get(f) in df.columns:
                X_df_columns.append(mapping[f])
            else:
                X_df_columns.append(df.columns[0])
        X = df[X_df_columns].copy()
        X.columns = feature_names
    else:
        X = df[feature_names].copy()

    #& Coerce data types to float/int
    print("Coercing all feature data types to clean numeric values for LightGBM...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    #? Strict 20% validation split to eliminate training memorization
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    #~ base sensors (not mean or deviations)
    base_sensor_keywords = ["temperature", "speed", "Torque", "wear", "temp", "rpm"]
    target_noise_columns = [
        col for col in X_val.columns 
        if any(keyword.lower() in col.lower() for keyword in base_sensor_keywords)
    ]
    
    print(f"Dynamic Leak Protection: Injecting noise into all {len(target_noise_columns)} cascading sensor features...")

    #todo Setup stress-testing noise tiers
    noise_tiers = {
        "Baseline (Unseen 0%)": 0.0,
        "Low Distortion (1%)": 0.01,
        "Moderate Jitter (5%)": 0.05,
        "Severe Corruption (10%)": 0.10
    }
    
    performance_records = []
    
    print("\nCommencing Unseen Data Stress Test Loops...")
    for tier_name, noise_level in noise_tiers.items():
        if noise_level > 0 and len(target_noise_columns) > 0:
            #& inject gaussian noise
            X_test_corrupted = inject_gaussian_noise(X_val, target_noise_columns, noise_level=noise_level)
        else:
            X_test_corrupted = X_val.copy()
            
        #~ perfect feature tracking order
        X_test_corrupted = X_test_corrupted[feature_names]
        
        #! Safe mathematical probability classification
        probabilities = model.predict_proba(X_test_corrupted)[:, 1]
        predictions = (probabilities >= baseline_threshold).astype(int)
        
        metrics = {
            "Tier": tier_name,
            "Noise Level": f"{noise_level*100}%",
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1-Score": f1_score(y_val, predictions, zero_division=0)
        }
        performance_records.append(metrics)
        
    
    report_df = pd.DataFrame(performance_records)
    
    print("\n" + "="*70)
    print("UNSEEN DATA NOISE SENSITIVITY REPORT CARD")
    print("="*70)
    print(report_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format
    }))
    print("="*70)
    
    report_df.to_csv(BASE_DIR / "Dataset" / "noise_sensitivity_results.csv", index=False)
    print("Analysis results successfully saved to 'noise_sensitivity_results.csv'")

if __name__ == "__main__":

    BASE_DIR = Path().resolve()
    DATA_PATH = BASE_DIR / "Dataset" / "predictive_maintenance_master_features.csv"
    MODEL_PATH = BASE_DIR / "models" / "lgbm_maintenance_model.pkl"
    
    run_leak_proof_robustness_test(DATA_PATH, MODEL_PATH)