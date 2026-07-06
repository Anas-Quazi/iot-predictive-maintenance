import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.stress_tester import inject_gaussian_noise
from pathlib import Path

def plot_noise_pr_curves(data_path: str, model_artifact_path: str, output_image_path: str = "noise_pr_curves.png"):
    print("Loading assets for Precision-Recall stress profiling...")
    payload = joblib.load(model_artifact_path)
    model = payload["model"]
    baseline_threshold = payload["optimal_threshold"]
    feature_names = payload["feature_names"]
    
    df = pd.read_csv(data_path)
    target_col = "Machine failure" if "Machine failure" in df.columns else "Machine_failure"
    y = df[target_col]
    
    # Structural column header alignment
    if not all(col in df.columns for col in feature_names):
        mapping = {f: f.replace('_', ' ').replace(' K', ' [K]').replace(' rpm', ' [rpm]').replace(' Nm', ' [Nm]').replace(' min', ' [min]') for f in feature_names}
        X_df_columns = [f if f in df.columns else (mapping[f] if mapping.get(f) in df.columns else df.columns[0]) for f in feature_names]
        X = df[X_df_columns].copy()
        X.columns = feature_names
    else:
        X = df[feature_names].copy()

    # Coerce to clean numeric types
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)
    
    # THE ADJUSTMENT: Use the full feature matrix to align perfectly with the 10,000 support rows
    X_full = X.copy()
    
    base_sensor_keywords = ["temperature", "speed", "Torque", "wear", "temp", "rpm"]
    target_noise_columns = [col for col in X_full.columns if any(k.lower() in col.lower() for k in base_sensor_keywords)]
    
    noise_tiers = {
        "Baseline (0% Noise)": (0.0, "blue"),
        "Low Distortion (1%)": (0.01, "orange"),
        "Moderate Jitter (5%)": (0.05, "green"),
        "Severe Corruption (10%)": (0.10, "red")
    }
    
    plt.figure(figsize=(10, 7))
    
    print("Generating PR Curves across full 10,000 row tracking space...")
    for label, (noise_level, color) in noise_tiers.items():
        if noise_level > 0:
            X_test_corrupted = inject_gaussian_noise(X_full, target_noise_columns, noise_level=noise_level)
        else:
            X_test_corrupted = X_full.copy()
            
        X_test_corrupted = X_test_corrupted[feature_names]
        
        # Raw probability evaluation
        probabilities = model.predict_proba(X_test_corrupted)[:, 1]
        
        # Generate full curve trajectories
        precisions, recalls, thresholds = precision_recall_curve(y, probabilities)
        plt.plot(recalls, precisions, label=f"{label}", color=color, lw=2)
        
    # Mark the exact operational threshold line from your image
    plt.axvline(x=0.83, color="purple", linestyle="--", alpha=0.6, label=f"Calibrated Baseline Recall (Thresh={baseline_threshold:.4f})")
    
    plt.title("Precision-Recall Curve Behavior Under Sensor Noise (Full Support Matrix)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall (Failure Detection Rate)", fontsize=12)
    plt.ylabel("Precision (Alert Accuracy Rate)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Success! Precision-Recall comparison graph saved to root as '{output_image_path}'")

if __name__ == "__main__":

    BASE_DIR = Path().resolve()
    DATA_PATH = BASE_DIR / "Dataset" / "predictive_maintenance_master_features.csv"
    MODEL_PATH = BASE_DIR / "models" / "lgbm_maintenance_model.pkl"
    
    plot_noise_pr_curves(DATA_PATH, MODEL_PATH)