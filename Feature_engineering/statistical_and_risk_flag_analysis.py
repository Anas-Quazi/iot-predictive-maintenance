from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / 'Dataset' / 'ai4i2020_cleaned.csv'
df = pd.read_csv(csv_path)

# Working Copy
df_feat = df.copy()

def create_statistical_features(df_feat):
    """
    Creates Z-Score statistical features on top of physical features.
    Requires: power, temp_diff already in df (from physics features).

    Features created:
        rpm_zscore, torque_zscore, power_zscore,
        wear_zscore, temp_diff_zscore

    Returns: df with new statistical feature columns added
    """

    df_feat['rpm_zscore']       = (df_feat['rpm']       - df_feat['rpm'].mean())       / df_feat['rpm'].std()
    df_feat['torque_zscore']    = (df_feat['torque']    - df_feat['torque'].mean())    / df_feat['torque'].std()
    df_feat['power_zscore']     = (df_feat['power']     - df_feat['power'].mean())     / df_feat['power'].std()
    df_feat['wear_zscore']      = (df_feat['tool_wear'] - df_feat['tool_wear'].mean()) / df_feat['tool_wear'].std()
    df_feat['temp_diff_zscore'] = (df_feat['temp_diff'] - df_feat['temp_diff'].mean()) / df_feat['temp_diff'].std()

    zscore_cols = ['rpm_zscore', 'torque_zscore', 'power_zscore', 'wear_zscore', 'temp_diff_zscore']

    print("=" * 60)
    print("Z-SCORE FEATURES CREATED")
    print("=" * 60)
    print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Extreme (>3σ)':>15}")
    print("  " + "-" * 58)
    for col in zscore_cols:
        extreme = (df_feat[col].abs() > 3).sum()
        print(f"  {col:<25} {df_feat[col].mean():>8.2f} {df_feat[col].std():>8.2f} {extreme:>15}")

    return df_feat      


def create_risk_flag_features(df_feat):
    """
    Creates binary general risk flag features based on threshold conditions.
    Requires: power, temp_diff already in df (from physics features).

    Flags created:
        high_wear_flag     — 1 if tool_wear > 200
        high_torque_flag   — 1 if torque > 60
        low_rpm_flag       — 1 if rpm < 1400
        high_temp_flag     — 1 if temp_diff > 15
        power_anomaly_flag — 1 if power is beyond mean ± 2std

    Returns: df with new risk flag columns added
    """

    # General Risk Flags
    df_feat['high_wear_flag']   = (df_feat['tool_wear'] > 200).astype(int)
    df_feat['high_torque_flag'] = (df_feat['torque'] > 60).astype(int)
    df_feat['low_rpm_flag']     = (df_feat['rpm'] < 1400).astype(int)
    df_feat['high_temp_flag']   = (df_feat['temp_diff'] > 15).astype(int)

    power_mean = df_feat['power'].mean()
    power_std  = df_feat['power'].std()
    df_feat['power_anomaly_flag'] = (
        (df_feat['power'] < power_mean - 2 * power_std) |
        (df_feat['power'] > power_mean + 2 * power_std)
    ).astype(int)

    flag_cols = [
        'high_wear_flag', 'high_torque_flag', 'low_rpm_flag',
        'high_temp_flag', 'power_anomaly_flag'
    ]

    print("=" * 60)
    print("RISK FLAG FEATURES CREATED")
    print("=" * 60)
    print(f"  {'Flag':<25} {'Flagged':>8} {'% Data':>8} {'Actual Fail':>12} {'Precision':>10}")
    print("  " + "-" * 66)
    for col in flag_cols:
        count     = df_feat[col].sum()
        pct       = count / len(df_feat) * 100
        actual    = df_feat[df_feat[col] == 1]['Machine failure'].sum()
        precision = actual / count * 100 if count > 0 else 0
        print(f"  {col:<25} {count:>8} {pct:>7.1f}% {actual:>12} {precision:>9.0f}%")

    return df_feat  
