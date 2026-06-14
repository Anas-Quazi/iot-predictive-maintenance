import pandas as pd
import numpy as np
from pathlib import Path

#* Polynomial Features
def create_polynomial_features(df_feat):

    """
    Function to create polynomial features based on existing numerical features in the dataset
    """

    df_feat['torque_squared'] = df_feat['Torque [Nm]'] ** 2
    df_feat['wear_squared'] = df_feat['Tool wear [min]'] ** 2
    df_feat['power_squared'] = df_feat['power'] ** 2
    df_feat['temp_diff_squared'] = df_feat['temp_diff'] ** 2
    df_feat['sqrt_power'] = np.sqrt(df_feat['power'])
    df_feat['sqrt_wear'] = np.sqrt(df_feat['Tool wear [min]'])
    df_feat['log_power'] = np.log1p(df_feat['power'])
    df_feat['log_strain'] = np.log1p(df_feat['strain_index'])

    print('Polynomial features created succesfully!')
    df_feat[['torque_squared', 'wear_squared', 'power_squared', 'temp_diff_squared', 'sqrt_power', 'sqrt_wear', 'log_power', 'log_strain']].describe()

    return df_feat


#& failure proximity feature (how much machine is close to failure threshold)
def create_failure_proximity_features(df_feat):

    """
    Function to create failure proximity feature that tells how close a machine is to each failure threshold 
    """

    osf_threshold = df_feat['Type'].map({0 : 11000, 1 : 12000, 2 : 13000})

    df_feat['distance_to_twf'] = 200 - df_feat['Tool wear [min]']
    df_feat['distance_to_hdf_temp'] = df_feat['temp_diff'] - 8.6
    df_feat['distance_to_hdf_rpm'] = df_feat['Rotational speed [rpm]'] - 1380
    df_feat['distance_to_pwf_lower'] = df_feat['power'] - 40000
    df_feat['distance_to_pwf_upper'] = 80000 - df_feat['power'] 
    df_feat['distance_to_osf'] = osf_threshold - df_feat['strain_index']
    df_feat['risk_score'] = (df_feat['hdf_risk_flag'] + df_feat['pwf_risk_flag'] + df_feat['osf_risk_flag'] + df_feat['twf_risk_flag'])

    print('Failure proximity features created succesfully!')
    return df_feat

if __name__ == '__main__':

    BASE_DIR = Path().resolve().parent
    df = pd.read_csv(BASE_DIR / 'IoT-Predictive-Maintenance' / 'Dataset' / 'ai4i2020_cleaned.csv')

    #? create copy to prevent override
    df_feat = df.copy()

    #^ recreate derived features needed for new ones
    df_feat['power'] = df_feat['Torque [Nm]'] * df_feat['Rotational speed [rpm]']
    df_feat['temp_diff'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    df_feat['strain_index'] = df_feat['Torque [Nm]'] * df_feat['Tool wear [min]']

    df_feat['hdf_risk_flag'] = ((df_feat['temp_diff'] < 8.6) & 
                            (df_feat['Rotational speed [rpm]'] < 1380)).astype(int)
    df_feat['pwf_risk_flag'] = ((df_feat['power'] < 40000) | 
                            (df_feat['power'] > 80000)).astype(int)
    osf_threshold = df_feat['Type'].map({0: 11000, 1: 12000, 2: 13000})
    df_feat['osf_risk_flag'] = (df_feat['strain_index'] > osf_threshold).astype(int)
    df_feat['twf_risk_flag'] = (df_feat['Tool wear [min]'] > 200).astype(int)
    
    df_feat = create_polynomial_features(df_feat)
    df_feat = create_failure_proximity_features(df_feat)
    
    print("Done! New features:")
    new = ['torque_squared','wear_squared','power_squared','temp_diff_squared',
           'sqrt_power','sqrt_wear','log_power','log_strain',
           'distance_to_twf','distance_to_hdf_temp','distance_to_hdf_rpm',
           'distance_to_pwf_lower','distance_to_pwf_upper','distance_to_osf','risk_score']
    print(df_feat[new].head())
    