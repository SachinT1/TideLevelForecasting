

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import config 

# Import your new physics engine
from harmonics import solve_harmonics

def run_preprocessing():
    print(f"--- Step 1: Loading Data from {config.INPUT_CSV} ---")
    df = pd.read_csv(config.INPUT_CSV, names=['time', 'observed_level'])
    df['time'] = pd.to_datetime(df['time'], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['time']).sort_values(by='time').set_index('time')

    print("--- Step 2: Extracting the Largest Pristine Chunk ---")
    df_hourly = df.resample('h').mean()
    mask = df_hourly['observed_level'].isna()
    block_ids = mask.cumsum()
    valid_data = df_hourly[~mask]
    longest_block_id = valid_data.groupby(block_ids).size().idxmax()
    df_pristine = df_hourly[block_ids == longest_block_id].dropna().copy()

    print("--- Step 3: Generating y_phy (Physics Engine) ---")
    # Pass the numpy array directly to your module
    # verbose=True will print the M2/S2 amplitudes to your console!
    df_pristine['s_tide_pred'] = solve_harmonics(df_pristine['observed_level'].values, verbose=True)

    print("--- Step 4: Normalization & Saving Scaler ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    cols = ['observed_level', 's_tide_pred']
    df_pristine[cols] = scaler.fit_transform(df_pristine[cols])
    joblib.dump(scaler, config.SCALER_PATH)

    print("--- Step 5: Saving Processed Data ---")
    df_pristine.reset_index().to_csv(config.PROCESSED_CSV, index=False)

