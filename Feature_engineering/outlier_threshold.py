import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy import stats

# Load dataset
BASE_DIR = Path().resolve().parent
df = pd.read_csv(BASE_DIR / 'Dataset' / 'ai4i2020_cleaned.csv')

# working copy
df_feat = df.copy()

# Recreate derived features needed for threshold flags
df_feat['power']        = df_feat['Torque [Nm]'] * df_feat['Rotational speed [rpm]']
df_feat['temp_diff']    = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
df_feat['strain_index'] = df_feat['Torque [Nm]'] * df_feat['Tool wear [min]']

print("Shape:", df_feat.shape)
print("Derived features ready: power, temp_diff, strain_index")

#Outlier Analysis
def outlier_analysis(df_feat):
    
    # ── IQR ──────────────────────────────────────────────────────────────────
    def iqr_outlier_flag(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).astype(int)

    df_feat['rpm_outlier_flag']    = iqr_outlier_flag(df_feat, 'Rotational speed [rpm]')
    df_feat['torque_outlier_flag'] = iqr_outlier_flag(df_feat, 'Torque [Nm]')

    print("=== IQR Method ===")
    for col, flag in [('Rotational speed [rpm]', 'rpm_outlier_flag'),
                      ('Torque [Nm]',            'torque_outlier_flag')]:
        out  = df_feat[df_feat[flag] == 1]['Machine failure'].mean() * 100
        norm = df_feat[df_feat[flag] == 0]['Machine failure'].mean() * 100
        print(f'{col}')
        print(f'  Outlier rows failure rate : {out:.2f}%')
        print(f'  Normal  rows failure rate : {norm:.2f}%\n')

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"{'Feature':<30} {'IQR':>6}")
    print("-" * 40)
    for col, iqr_flag in [
        ('Rotational speed [rpm]', 'rpm_outlier_flag'),
        ('Torque [Nm]',            'torque_outlier_flag')
    ]:
        print(f"{col:<30} {df_feat[iqr_flag].sum():>6} ")
    print("\n")
    return df_feat

#Threshold Bsed Risk Feature
def threshold_risk_flags(df_feat):

    # ── HDF: Heat Dissipation Failure ─────────────────────────────────────────
    df_feat['hdf_risk_flag'] = (
        (df_feat['temp_diff'] < 8.6) &
        (df_feat['Rotational speed [rpm]'] < 1380)
    ).astype(int)

    # ── PWF: Power Failure ────────────────────────────────────────────────────
    df_feat['pwf_risk_flag'] = (
        (df_feat['power'] < 40000) |
        (df_feat['power'] > 80000)
    ).astype(int)

    # ── OSF: Overstrain Failure ───────────────────────────────────────────────
    df_feat['osf_risk_flag'] = (
        ((df_feat['Type'] == 0) & (df_feat['strain_index'] > 11000)) |  # L
        ((df_feat['Type'] == 1) & (df_feat['strain_index'] > 12000)) |  # M
        ((df_feat['Type'] == 2) & (df_feat['strain_index'] > 13000))    # H
    ).astype(int)

    # ── TWF: Tool Wear Failure ────────────────────────────────────────────────
    df_feat['twf_risk_flag'] = (df_feat['Tool wear [min]'] >= 200).astype(int)

    print("=== Threshold Based Flags ===")
    print(f"  hdf_risk_flag : {df_feat['hdf_risk_flag'].sum()}")
    print(f"  pwf_risk_flag : {df_feat['pwf_risk_flag'].sum()}")
    print(f"  osf_risk_flag : {df_feat['osf_risk_flag'].sum()}")
    print(f"  twf_risk_flag : {df_feat['twf_risk_flag'].sum()}")
    return df_feat
