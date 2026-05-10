
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import config
from harmonics import TIDAL_CONSTITUENTS
from model import PhyBiGRU 

def get_harmonic_weights(elevation_array):
    t = np.arange(len(elevation_array), dtype=float)
    A = [np.ones(len(t))] 
    for name, freq in TIDAL_CONSTITUENTS.items():
        A.append(np.cos(2 * np.pi * freq * t))
        A.append(np.sin(2 * np.pi * freq * t))
    A = np.column_stack(A)
    weights, _, _, _ = np.linalg.lstsq(A, elevation_array, rcond=-1)
    return weights

def project_future_physics(start_hour_index, forecast_hours, weights):
    t_future = np.arange(start_hour_index, start_hour_index + forecast_hours, dtype=float)
    A_future = [np.ones(len(t_future))]
    for name, freq in TIDAL_CONSTITUENTS.items():
        A_future.append(np.cos(2 * np.pi * freq * t_future))
        A_future.append(np.sin(2 * np.pi * freq * t_future))
    A_future = np.column_stack(A_future)
    return A_future @ weights

def run_out_of_sample_backtest(backtest_days=5, window_size=24):
    forecast_hours = backtest_days * 24
    print(f"--- Initiating {backtest_days}-Day Out-of-Sample Backtest ---")

    # 1. Load Data, Scaler, and Model
    df = pd.read_csv(config.PROCESSED_CSV)
    scaler = joblib.load(config.SCALER_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    model = PhyBiGRU(input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(config.TRAINED_MODEL, map_location=device))
    model.eval()

    # 2. Split Data
    history_df = df.iloc[:-forecast_hours].copy()
    ground_truth_df = df.iloc[-forecast_hours:].copy()
    total_historical_hours = len(history_df)

    # 3. Physics Setup
    unscaled_history = scaler.inverse_transform(history_df[['observed_level', 's_tide_pred']])
    real_historical_observed = unscaled_history[:, 0]
    weights = get_harmonic_weights(real_historical_observed)
    future_y_phy_real = project_future_physics(total_historical_hours, forecast_hours, weights)

    # Scale the future physics
    dummy_df = pd.DataFrame({
        'observed_level': np.zeros_like(future_y_phy_real),
        's_tide_pred': future_y_phy_real
    })
    future_scaled = scaler.transform(dummy_df)
    future_y_phy_scaled = future_scaled[:, 1]

    last_window = history_df[['observed_level', 's_tide_pred']].tail(window_size).values 
    current_seq = torch.FloatTensor(last_window).unsqueeze(0).to(device) 

    predictions_scaled = []

    print("Running Autoregressive Inference with Physics Guiderails...")
    with torch.no_grad():
        for i in range(forecast_hours):
            # Predict next hour
            pred = model(current_seq).item()
            predictions_scaled.append(pred)
            
            
            next_step_data = np.array([[pred, future_y_phy_scaled[i]]])
            next_step = torch.FloatTensor(next_step_data).unsqueeze(0).to(device) # Shape: (1, 1, 2)
            
            # Slide the window
            current_seq = torch.cat((current_seq[:, 1:, :], next_step), dim=1)

    

    # 6. Inverse Transform Results
    pred_array = np.array(predictions_scaled)
    final_output_scaled = np.column_stack([pred_array, future_y_phy_scaled])
    final_output_real = scaler.inverse_transform(final_output_scaled)
    
    forecast_levels = final_output_real[:, 0]
    forecast_physics = final_output_real[:, 1]

    # 7. Ground Truth Inverse
    unscaled_truth = scaler.inverse_transform(ground_truth_df[['observed_level', 's_tide_pred']])
    actual_hidden_levels = unscaled_truth[:, 0]

    # 8. Plot
    plot_backtest(real_historical_observed, forecast_levels, forecast_physics, actual_hidden_levels, forecast_hours)

def plot_backtest(historical_real, forecast_levels, forecast_physics, actual_hidden_levels, forecast_hours):
    context_hours = 72 # Show 3 days of history for visual context
    history_plot = historical_real[-context_hours:]
    
    t_history = np.arange(-context_hours, 0)
    t_forecast = np.arange(0, forecast_hours)
    
    plt.figure(figsize=(16, 7))
    
    # Plot Past History
    plt.plot(t_history, history_plot, color='gray', label='Past Context (Seen Data)', linewidth=1.5)
    
    # Plot The "Hidden" Ground Truth
    plt.plot(t_forecast, actual_hidden_levels, color='black', label='Actual Ground Truth (Hidden from Model)', linewidth=2.5)
    
    # Plot The Physics Baseline
    plt.plot(t_forecast, forecast_physics, color='blue', linestyle='--', label='S_Tide Physical Baseline ($y_{phy}$)', alpha=0.7)
    
    # Plot The Model's Forecast
    plt.plot(t_forecast, forecast_levels, color='red', label='BiGRU Recursive Forecast', linewidth=2)
    
    plt.axvline(x=0, color='gray', linestyle='-.', label='Forecast Start ($t=0$)')
    
    plt.title(f"Out-of-Sample Backtest: {forecast_hours//24}-Day Recursive Forecast vs. Ground Truth")
    plt.xlabel("Time (Hours relative to Forecast Start)")
    plt.ylabel("Water Level (Meters)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prof_backtest_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_out_of_sample_backtest(backtest_days=5)

