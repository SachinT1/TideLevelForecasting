
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os

def plot_time_series(y_true, y_pred, y_phy, save_path="/kaggle/working/fig_timeseries.png"):
    """Recreates Figure 6/10: Time-series comparison of a subset of predictions."""
    plt.figure(figsize=(14, 6))
    
    # Plot a subset (e.g., 300 hours) so it's readable
    subset_size = min(300, len(y_true))
    
    plt.plot(y_true[:subset_size], label='Actual Tide Level', color='black', linewidth=2)
    plt.plot(y_phy[:subset_size], label='S_TIDE (y_phy)', color='blue', linestyle='--', alpha=0.7)
    plt.plot(y_pred[:subset_size], label='Phy-BiGRU Forecast', color='red', linestyle='-.')
    
    plt.title('Tide Level Forecast Comparison (Unseen Test Data)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Water Level (Meters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Time-Series plot to {save_path}")

def plot_scatter(y_true, y_pred, save_path="/kaggle/working/fig_scatter.png"):
    """Recreates Figure 7/11: Scatter plot showing fitting accuracy and R2."""
    plt.figure(figsize=(8, 8))
    
    r2 = r2_score(y_true, y_pred)
    
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolors='k', label=f'Predictions (R²={r2:.4f})')
    
    # Plot the 1:1 ideal diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ideal Fit')
    
    plt.title('Prediction vs. Actual (Scatter)')
    plt.xlabel('Actual Tide Level (Meters)')
    plt.ylabel('Predicted Tide Level (Meters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Scatter plot to {save_path}")

def plot_error_dist(y_true, y_pred, save_path="/kaggle/working/fig_error_dist.png"):
    """Plots a histogram of the residuals to check for bias."""
    plt.figure(figsize=(10, 5))
    
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.title('Error Distribution (Residuals in Meters)')
    plt.xlabel('Error (Prediction - Actual)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Error Distribution plot to {save_path}")

def generate_all_plots(y_true, y_pred, y_phy):
    """Wrapper function to generate and save all plots."""
    plot_time_series(y_true, y_pred, y_phy)
    plot_scatter(y_true, y_pred)
    plot_error_dist(y_true, y_pred)