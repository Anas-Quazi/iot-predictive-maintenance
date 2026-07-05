import numpy as np
import pandas as pd

def inject_gaussian_noise(df: pd.DataFrame, target_columns: list, noise_level: float, random_state: int = 42) -> pd.DataFrame:
    """
    Deliberately corrupts specified continuous features with Gaussian White Noise 
    scaled proportionally to each feature's natural standard deviation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The original, pristine dataset or validation slice.
    target_columns : list
        List of continuous sensor column names to inject noise into.
    noise_level : float
        The percentage of standard deviation to inject (e.g., 0.01 for 1%, 0.05 for 5%).
    random_state : int
        Ensures reproducible random noise generation.
        
    Returns:
    --------
    pd.DataFrame
        A deep copy of the dataframe containing the corrupted sensor telemetry.
    """
    np.random.seed(random_state)
    corrupted_df = df.copy()
    
    for col in target_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the provided DataFrame.")
            
        # 1. Calculate the feature's natural standard deviation
        col_std = df[col].std()
        
        # 2. Scale the noise standard deviation based on the targeted tier
        noise_std = noise_level * col_std
        
        # 3. Generate a normal distribution of noise matching the column's length
        gaussian_noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df))
        
        # 4. Inject the noise into the copied dataframe
        corrupted_df[col] = corrupted_df[col] + gaussian_noise
        
    return corrupted_df

if __name__ == "__main__":
    print("Initializing Stress Testing Module Verification...")
    
    # Quick sanity check using dummy telemetry data
    sample_data = pd.DataFrame({
        "Rotational_speed_rpm": [1350.0, 1500.0, 1650.0, 1400.0, 1550.0],
        "Torque_Nm": [52.4, 40.2, 38.1, 49.8, 42.0],
        "Type": ["M", "L", "L", "M", "H"]  # Categorical columns must stay untouched
    })
    
    continuous_sensors = ["Rotational_speed_rpm", "Torque_Nm"]
    
    print("\n--- Baseline Telemetry ---")
    print(sample_data[continuous_sensors].head(2))
    
    # Test a 5% moderate degradation injection
    corrupted_sample = inject_gaussian_noise(sample_data, continuous_sensors, noise_level=0.05)
    
    print("\n--- Corrupted Telemetry (5% Noise Injected) ---")
    print(corrupted_sample[continuous_sensors].head(2))
    print("\nNoise Injection Engine verification successful.")