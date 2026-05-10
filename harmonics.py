
import numpy as np

# Standard tidal frequencies (cycles per hour) from NOAA / UTide references
TIDAL_CONSTITUENTS = {
    'M2': 1/12.4206012,  # Principal lunar semidiurnal
    'S2': 1/12.0000000,  # Principal solar semidiurnal
    'N2': 1/12.6583482,  # Larger lunar elliptic semidiurnal
    'K2': 1/11.9672348,  # Lunisolar semidiurnal
    'K1': 1/23.9344722,  # Lunar diurnal
    'O1': 1/25.8193387,  # Lunar diurnal
    'P1': 1/24.0658902,  # Solar diurnal
    'Q1': 1/26.8683567   # Larger lunar elliptic diurnal
}

def solve_harmonics(elevation_array, verbose=True):
    """
    Performs Ordinary Least Squares (OLS) harmonic analysis.
    Bypasses UTide/SVD convergence issues on Apple Silicon.
    
    Args:
        elevation_array (np.array): 1D array of hourly water levels.
        verbose (bool): If True, prints the calculated amplitudes.
        
    Returns:
        np.array: The reconstructed physical tidal wave (y_phy).
    """
    t = np.arange(len(elevation_array), dtype=float)
    
    # 1. Build the Design Matrix (A)
    # Start with a column of ones for the Mean Sea Level (Z0)
    A = [np.ones(len(t))] 
    
    # Add a cosine and sine column for each frequency
    names = []
    for name, freq in TIDAL_CONSTITUENTS.items():
        A.append(np.cos(2 * np.pi * freq * t))
        A.append(np.sin(2 * np.pi * freq * t))
        names.append(name)
        
    A = np.column_stack(A)
    
    # 2. Solve the Linear System
    # rcond=-1 ensures maximum stability across different CPU architectures
    weights, residuals, rank, s = np.linalg.lstsq(A, elevation_array, rcond=-1)
    
    # 3. Optional Debugging: Calculate and print amplitudes
    if verbose:
        z0 = weights[0]
        print(f"--- Harmonic Analysis Results ---")
        print(f"Mean Sea Level (Z0): {z0:.4f} meters")
        
        # Amplitudes are sqrt(cos_weight^2 + sin_weight^2)
        for i, name in enumerate(names):
            cos_w = weights[1 + (i*2)]
            sin_w = weights[2 + (i*2)]
            amplitude = np.sqrt(cos_w**2 + sin_w**2)
            print(f"  {name} Amplitude: {amplitude:.4f} m")
            
        if np.any(np.abs(weights) > 100):
            print("WARNING: Unusually high weights detected. Check for data gaps.")
    
    # 4. Reconstruct the physical baseline
    y_phy = A @ weights
    
    return y_phy