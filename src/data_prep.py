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

    #& 1. Setup target vector (y) — raw column names as they appear in CSV
    target_col = 'Machine failure'
    if target_col not in df.columns:
        raise KeyError(f"Critical error: target column '{target_col}' not found in dataset")
    y = df[target_col].copy()

    #! 2. Data leakage firewall — drop failure sub-type proxy columns
    #! and administrative ID columns that have zero predictive value
    proxy_targets = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    database_ids  = ['UDI', 'Product ID', 'Timestamp', 'Timestamp.1']

    exclude_from_features = proxy_targets + database_ids + [target_col]
    X = df.drop(columns=[col for col in exclude_from_features if col in df.columns])

    #& 3. Catch any leftover string/object columns (e.g. un-encoded categoricals)
    #& and drop them so the matrix stays fully numeric
    string_columns = X.select_dtypes(include=['object', 'str']).columns.tolist()
    if string_columns:
        print(f"warning: found unencoded text columns {string_columns}. dropping from X matrix.")
        X = X.drop(columns=string_columns)

    #todo normalize column header with regex
    X.columns = X.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    X.columns = X.columns.str.replace(r'_{2,}', '_', regex=True).str.strip('_')
    
    #! 4. Data cleanroom invariant audits — hard stops if anything is wrong
    assert not X.isnull().values.any(), \
        "pipeline guardrail tripped: NaN values detected in X matrix."
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes), \
        "pipeline guardrail tripped: non-numeric types still present in X matrix."

    print("phase 1 verification successful!")
    print(f"   -> feature shape : {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"   -> failure rate  : {y.mean() * 100:.2f}% ({y.sum()} total breakdowns)")
    print("-" * 70)

    return X, y


if __name__ == "__main__":
    try:
        X, y = load_and_sanitize_data("Dataset/predictive_maintenance_master_features.csv")
    except Exception as e:
        print(f"local test failed: {str(e)}") 
