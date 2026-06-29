import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def execute_sealed_cv_split(X, y, n_splits=5, random_state=42):
    """
    Perform Stratified K-Fold Cross-Validation with leakage-safe SMOTE.

    Each fold preserves the original class distribution using
    StratifiedKFold. SMOTE is applied only to the training partition,
    while the validation partition remains untouched to simulate
    real-world unseen data.

    Args:
        X (pd.DataFrame): Preprocessed feature matrix.
        y (pd.Series): Binary target labels.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Yields:
        tuple:
            X_train_resampled (pd.DataFrame): SMOTE-balanced training features.
            y_train_resampled (pd.Series): Balanced training labels.
            X_val (pd.DataFrame): Original validation features.
            y_val (pd.Series): Original validation labels.
    """

    print(f"Initializing {n_splits}-Fold Stratified Validation")

    # Preserve class distribution across all folds
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n Processing Fold {fold}")

        # Split into training and validation sets
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(
            f"Before SMOTE - Failure={y_train.sum()} | "
            f"Stable={len(y_train) - y_train.sum()}"
        )

        # Apply SMOTE only on the training data
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train
        )

        # Convert back to pandas objects
        X_train_resampled = pd.DataFrame(
            X_train_resampled,
            columns=X.columns
        )

        y_train_resampled = pd.Series(
            y_train_resampled,
            name=y.name
        )

        print(
            f"After SMOTE  - {X_train_resampled.shape[0]} balanced samples"
        )
        print(
            f"Validation - {X_val.shape[0]} untouched samples"
        )

        # Return one fold at a time
        yield (
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
        )


if __name__ == "__main__":
    from data_prep import load_and_sanitize_data

    try:
        X, y = load_and_sanitize_data(
            "Dataset/predictive_maintenance_master_features.csv"
        )

        fold_generator = execute_sealed_cv_split(X, y)

        X_tr, y_tr, X_va, y_va = next(fold_generator)

        print("\n Phase 2 verification successful!")

    except Exception as e:
        print(f" Verification failed: {e}")