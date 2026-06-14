import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_statistical_features(df):
    """
    Creates Z-Score statistical features on top of physical features.
    Requires: power, temp_diff already in df (from physics features).

    Features created:
        rpm_zscore, torque_zscore, power_zscore,
        wear_zscore, temp_diff_zscore

    Returns: df with new statistical feature columns added
    """

    df['rpm_zscore']       = (df['rpm']       - df['rpm'].mean())       / df['rpm'].std()
    df['torque_zscore']    = (df['torque']    - df['torque'].mean())    / df['torque'].std()
    df['power_zscore']     = (df['power']     - df['power'].mean())     / df['power'].std()
    df['wear_zscore']      = (df['tool_wear'] - df['tool_wear'].mean()) / df['tool_wear'].std()
    df['temp_diff_zscore'] = (df['temp_diff'] - df['temp_diff'].mean()) / df['temp_diff'].std()

    zscore_cols = ['rpm_zscore', 'torque_zscore', 'power_zscore', 'wear_zscore', 'temp_diff_zscore']

    print("=" * 60)
    print("Z-SCORE FEATURES CREATED")
    print("=" * 60)
    print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Extreme (>3σ)':>15}")
    print("  " + "-" * 58)
    for col in zscore_cols:
        extreme = (df[col].abs() > 3).sum()
        print(f"  {col:<25} {df[col].mean():>8.2f} {df[col].std():>8.2f} {extreme:>15}")

    return df


def create_risk_flag_features(df):
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
    df['high_wear_flag']   = (df['tool_wear'] > 200).astype(int)
    df['high_torque_flag'] = (df['torque'] > 60).astype(int)
    df['low_rpm_flag']     = (df['rpm'] < 1400).astype(int)
    df['high_temp_flag']   = (df['temp_diff'] > 15).astype(int)

    power_mean = df['power'].mean()
    power_std  = df['power'].std()
    df['power_anomaly_flag'] = (
        (df['power'] < power_mean - 2 * power_std) |
        (df['power'] > power_mean + 2 * power_std)
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
        count     = df[col].sum()
        pct       = count / len(df) * 100
        actual    = df[df[col] == 1]['Machine failure'].sum()
        precision = actual / count * 100 if count > 0 else 0
        print(f"  {col:<25} {count:>8} {pct:>7.1f}% {actual:>12} {precision:>9.0f}%")

    return df
