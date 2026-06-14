import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 2. Physics Features ────────────────────────────────────────────────────────
def create_physics_features(df_input):
    """
    Recreates power and temp_diff — required as inputs
    for interaction and machine type feature functions.
    Skip if chaining from the main feature engineering notebook.
    """
    df_feat = df_input.copy()

    df_feat['power'] = df_feat['Torque [Nm]'] * df_feat['Rotational speed [rpm]']
    df_feat['temp_diff'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    df_feat['wear_rate'] = df_feat['Tool wear [min]'] / (df_feat['Rotational speed [rpm]'] + 1)
    df_feat['torque_per_wear'] = df_feat['Torque [Nm]'] / (df_feat['Tool wear [min]'] + 1)
    df_feat['strain_index'] = df_feat['Torque [Nm]'] * df_feat['Tool wear [min]']
    df_feat['power_per_temp'] = df_feat['power'] / df_feat['Process temperature [K]']

    print("Physics features created!")
    print(df_feat[['power', 'temp_diff', 'wear_rate', 'torque_per_wear',
                    'strain_index', 'power_per_temp']].describe().round(2))

    return df_feat


# ── 3. Interaction Features ────────────────────────────────────────────────────
def create_interaction_features(df_input):
    """
    Captures combined effects between two variables that
    individually may not predict failure but together do.
    Requires: power, temp_diff (run create_physics_features first).
    """
    df_feat = df_input.copy()

    # High torque under heat = elevated stress
    df_feat['thermal_torque_stress'] = df_feat['Torque [Nm]'] * df_feat['Process temperature [K]']

    # Combined operational load on a worn tool
    df_feat['rotational_load_on_worn_tool'] = (
        df_feat['Rotational speed [rpm]']
        * df_feat['Torque [Nm]']
        * df_feat['Tool wear [min]']
    )

    # Mechanical energy absorbed by a degrading tool
    df_feat['power_absorbed_by_wear'] = df_feat['power'] * df_feat['Tool wear [min]']

    # Thermal stress on an already worn tool
    df_feat['thermal_stress_on_worn_tool'] = df_feat['temp_diff'] * df_feat['Tool wear [min]']

    # Speed-induced heat imbalance
    df_feat['speed_heat_imbalance'] = df_feat['Rotational speed [rpm]'] * df_feat['temp_diff']

    interaction_cols = [
        'thermal_torque_stress', 'rotational_load_on_worn_tool',
        'power_absorbed_by_wear', 'thermal_stress_on_worn_tool', 'speed_heat_imbalance'
    ]
    print("Interaction features created!")
    print(df_feat[interaction_cols].describe().round(2))

    return df_feat


# ── 4. Machine Type Features ───────────────────────────────────────────────────
"""
Creates per-machine-type features.
Type is ordinal encoded: L=0, M=1, H=2.
Within-type z-scores capture how abnormal a reading
is FOR its machine grade, not globally.
Requires: power, temp_diff (run create_physics_features first).
"""

def create_machine_type_features(df_input):
    """
    Creates machine type indicator features.
    Type is ordinal encoded: L=0, M=1, H=2.
    """
    df_feat = df_input.copy()

    df_feat['is_low_quality'] = (df_feat['Type'] == 0).astype(int)
    df_feat['is_high_quality'] = (df_feat['Type'] == 2).astype(int)

    print("Machine type features created!")
    print(df_feat[['is_low_quality', 'is_high_quality']].head())

    return df_feat


# ── Pipeline ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = create_physics_features(df)
    df = create_interaction_features(df)
    df = create_machine_type_features(df)
