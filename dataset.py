

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import config

class TideDataset(Dataset):
    def __init__(self, csv_file, seq_length, forecast_horizon):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        
        self.features = self.data[['observed_level', 's_tide_pred']].values
        
        # We still keep these for the targets
        self.observed = self.data['observed_level'].values
        self.s_tide = self.data['s_tide_pred'].values
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon + 1
        
    def __getitem__(self, idx):
        
        # Slice shape: (seq_length, 2)
        x = self.features[idx : idx + self.seq_length]
        
        target_idx = idx + self.seq_length + self.forecast_horizon - 1
        y_true = self.observed[target_idx]
        y_phy = self.s_tide[target_idx]
        
        
        x_tensor = torch.tensor(x, dtype=torch.float32) 
        y_true_tensor = torch.tensor([y_true], dtype=torch.float32)
        y_phy_tensor = torch.tensor([y_phy], dtype=torch.float32)
        
        return x_tensor, y_true_tensor, y_phy_tensor

def get_chronological_dataloaders():
    dataset = TideDataset(config.PROCESSED_CSV, config.SEQ_LENGTH, config.FORECAST_HORIZON)
    total_size = len(dataset)
    train_end = int(0.8 * total_size)
    val_end = int(0.9 * total_size)
    
    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, total_size))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_dataset
