
import os
INPUT_CSV = "/data/hazira_data.csv" 

PROCESSED_CSV = "data/tide_data_processed.csv"

SCALER_PATH = "/data/scaler.pkl"
DB_PATH = "/data/phy_bigru_study.db"
BEST_PARAMS_PATH = "/data/best_params.json"
MODEL_SAVE_PREFIX = "/data/best_model_trial_"
TRAINED_MODEL = 'best_phy_bigru_final.pth'

# ==========================================
# 2. TIME-SERIES & SEQUENCE SETTINGS
# ==========================================
SEQ_LENGTH = 24       # Use past 24 hours
FORECAST_HORIZON = 1  # Predict 1 hour ahead

# ==========================================
# 3. PHY-BIGRU UNIFIED CONFIGURATION
# ==========================================
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_LAYERS = 3

# ==========================================
# 4. TRAINING & OPTIMIZATION SETTINGS
# ==========================================
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 20
N_TRIALS = 30         # Number of Bayesian Optimization iterations

# Optuna Search Space bounds (from the paper)
LAMBDA_MIN = 0.95
LAMBDA_MAX = 1.0
LR_MIN = 1e-5
LR_MAX = 1e-2
