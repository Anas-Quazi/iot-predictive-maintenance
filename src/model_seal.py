import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, accuracy_score
from data_prep import load_and_sanitize_data
from cv_engine import execute_sealed_cv_split
from train_lgbm import get_regularized_lgbm_model

def calibrate_and_seal_pipeline(data_path, model_output_path="models/lgbm_maintenance_model.pkl"):
    """
    Executes the full cross-validation loop, optimizes the decision threshold 
    to balance precision and recall, trains the final model, and serializes it.
    """
    #^ Load data assets
    X, y = load_and_sanitize_data(data_path)
    
    oof_predictions = np.zeros(len(X))
    fold_generator = execute_sealed_cv_split(X, y, n_splits=5)
    
    print("\nStarting Cross-Validation Evaluation Loop...")
    
    #todo Evaluate model stability across folds
    for fold, (X_train, y_train, X_val, y_val) in enumerate(fold_generator, 1):
        model = get_regularized_lgbm_model()
        model.fit(X_train, y_train)
        
        #? Predict probabilities for the validation fold (Index 1 is the failure class)
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_predictions[X_val.index] = val_probs
        print(f"   -> Fold {fold} Evaluation Completed.")

    #* Best threshold
    precision, recall, thresholds = precision_recall_curve(y, oof_predictions)
    
    #todo Calculate F1-score for each threshold to find the optimal mathematical balance
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    
    print("\n Metric Calibration Results:")
    print(f"   -> Optimal Operational Threshold: {optimal_threshold:.4f}")
    
    #& Apply calibrated threshold to generate final metrics
    calibrated_preds = (oof_predictions >= optimal_threshold).astype(int)
    print("\n Out-of-Fold Classification Summary:")
    print(f"Absolute Raw Accuracy: {accuracy_score(y, calibrated_preds):.6f}")
    print(classification_report(y, calibrated_preds, target_names=["Stable", "Failure"]))

    #^ Final Train and Serialization
    print("\n Sealing final model artifact on complete dataset...")
    final_model = get_regularized_lgbm_model()
    
    #~ Apply SMOTE to the entire dataset now that evaluation is complete
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    final_model.fit(X_resampled, y_resampled)
    
    #* Pack the model along with its metadata and operational threshold
    artifact_payload = {
        "model": final_model,
        "optimal_threshold": optimal_threshold,
        "feature_names": list(X.columns)
    }
    
    #? Save production artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(artifact_payload, model_output_path)
    print(f"Production artifact successfully sealed and saved to: {model_output_path}")

if __name__ == "__main__":
    calibrate_and_seal_pipeline("Dataset/predictive_maintenance_master_features.csv")