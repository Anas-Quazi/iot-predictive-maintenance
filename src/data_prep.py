import numpy as np
import pandas as pd

#* prepare data with interrogation and feature sanitization
def load_and_sanitize_data(file_path):

    """
    Ingest features from Delta Dataset, clean column header, purge target proxies, hide data leakage flags and returns sanitized X (features) and y (target) arrays for production modeling

    Parameters: 
        file path (str): path of Delta Dataset (predictive_maintenance_master_features.csv)

    Returns: 
        X (pd.DataFrame): Pure numeric operational feature matrix.
        y (pd.Series): Clean binary target array for machine failure.
    """

    print(f"ingesting data from {file_path}")
    df = pd.read_csv(file_path)

    #! 1. Normalize colmns for lightgbm
    #! replace space, brackets with underscore
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace(r'_{2,}', '_', regex=True).str.strip('_')

    #& 2. Setup target vector (y)
    target_col = 'Machine_failure'
    if target_col not in df.columns:
        raise KeyError(f"Critical error: target column {target_col} not found in dataset")
    y = df[target_col].copy()

    