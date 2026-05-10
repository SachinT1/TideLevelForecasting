import torch
import torch.optim as optim
import optuna
import joblib
import json
import os
import numpy as np


import config
from dataset import get_chronological_dataloaders
from model import PhyBiGRU, physics_informed_loss
import visualize

def objective(trial, train_loader, val_loader, device):
    """Optuna objective function for a single trial."""
    print(f"\n{'='*20} Starting Trial {trial.number + 1} {'='*20}")
    
    # 1. Hyperparameter Search Space
    lambda_weight = trial.suggest_float("lambda_weight", config.LAMBDA_MIN, config.LAMBDA_MAX)
    lr = trial.suggest_float("lr", config.LR_MIN, config.LR_MAX, log=True)
    
    # 2. Initialize Model & Optimizer
    
    model = PhyBiGRU(input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_rmse = float('inf')
    patience_counter = 0
    
    # 3. Training Loop
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        for x_b, y_t, y_p in train_loader:
            x_b, y_t, y_p = x_b.to(device), y_t.to(device), y_p.to(device)
            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = physics_informed_loss(y_pred, y_t, y_p, lambda_weight)
            loss.backward()
            optimizer.step()
            
        # 4. Validation
        model.eval()
        val_rmse = 0.0
        with torch.no_grad():
            
            for x_b, y_t, _ in val_loader:
                x_b, y_t = x_b.to(device), y_t.to(device)
                y_pred = model(x_b)
                val_rmse += torch.sqrt(torch.mean((y_t - y_pred)**2)).item()
        
        avg_val_rmse = val_rmse / len(val_loader)
        
        # 5. Early Stopping & Checkpointing
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            patience_counter = 0
            trial_model_path = f"{config.MODEL_SAVE_PREFIX}{trial.number}.pth"
            torch.save(model.state_dict(), trial_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break
                
    return best_val_rmse

def evaluate_best_model(best_trial_number, test_loader, device):
    print("\n--- Running Final Evaluation on Unseen Test Data ---")
    
    best_model_path = f"{config.MODEL_SAVE_PREFIX}{best_trial_number}.pth"
    model = PhyBiGRU(input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    predictions, actuals, physics_preds = [], [], []
    
    with torch.no_grad():
        for x_b, y_t, y_p in test_loader:
            x_b = x_b.to(device)
            y_pred = model(x_b)
            
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_t.cpu().numpy())
            physics_preds.extend(y_p.cpu().numpy())
            
    # 2. Inverse Transform
    scaler = joblib.load(config.SCALER_PATH)

    def manual_inverse(data_list, col_idx):
        data_arr = np.array(data_list).flatten()
        c_min = scaler.min_[col_idx]
        c_scale = scaler.scale_[col_idx]
        
        return (data_arr - c_min) / c_scale

    preds_meters = manual_inverse(predictions, 0)
    actuals_meters = manual_inverse(actuals, 0)
    physics_meters = manual_inverse(physics_preds, 1)
    
    final_rmse = np.sqrt(np.mean((actuals_meters - preds_meters)**2))
    print(f">>> FINAL PHYSICAL RMSE: {final_rmse:.4f} meters <<<")
    
    visualize.generate_all_plots(actuals_meters, preds_meters, physics_meters)

def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader, _ = get_chronological_dataloaders()
    
    study = optuna.create_study(
        study_name="phy_bigru_optimization", 
        storage=config.DB_PATH, 
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=config.N_TRIALS)
    
    with open(config.BEST_PARAMS_PATH, 'w') as f:
        json.dump(study.best_params, f)
        
    print(f"\nOptimization Finished. Best Trial: {study.best_trial.number}")
    evaluate_best_model(study.best_trial.number, test_loader, device)
if __name__ == "__main__":
    run_pipeline()
