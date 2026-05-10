

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import config 


from harmonics import solve_harmonics

def run_preprocessing():
    print(f"--- Step 1: Loading Data from {config.INPUT_CSV} ---")
    df = pd.read_csv(config.INPUT_CSV, names=['time', 'observed_level'])
    df['time'] = pd.to_datetime(df['time'], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['time']).sort_values(by='time').set_index('time')

    print("--- Step 2: Extracting the Largest Continuous Chunk ---")
    print("--- Step 2: Extracting the Largest Continuous Chunk ---")
    df_hourly = df.resample('h').mean()

    """ 
    GUIDE FOR DATA GAPS:
    The logic below performs 'Extraction', finding the single longest 
    uninterrupted sequence of data.
    
    If you want to 'piece together' chunks or fill small gaps, 
    COMMENT OUT the lines between START and END below, and use:
    
    df_hourly['observed_level'] = df_hourly['observed_level'].interpolate(method='linear', limit=3)
    df_pristine = df_hourly.dropna().copy()
    """

    # --- START OF EXTRACTION LOGIC ---
    mask = df_hourly['observed_level'].isna()
    block_ids = mask.cumsum()
    valid_data = df_hourly[~mask]
    
    # Identify the ID of the block with the most rows
    longest_block_id = valid_data.groupby(block_ids).size().idxmax()
    
    # Filter the dataframe to only include that specific continuous block
    df_pristine = df_hourly[block_ids == longest_block_id].dropna().copy()
    # --- END OF EXTRACTION LOGIC ---
    

    print("--- Step 3: Generating y_phy (Physics Engine) ---")
    
    df_pristine['s_tide_pred'] = solve_harmonics(df_pristine['observed_level'].values, verbose=True)

    print("--- Step 4: Normalization & Saving Scaler ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    cols = ['observed_level', 's_tide_pred']
    df_pristine[cols] = scaler.fit_transform(df_pristine[cols])
    joblib.dump(scaler, config.SCALER_PATH)

    print("--- Step 5: Saving Processed Data ---")
    df_pristine.reset_index().to_csv(config.PROCESSED_CSV, index=False)

