
import torch
import torch.nn as nn

class PhyBiGRU(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=3):
        super(PhyBiGRU, self).__init__()
        
        # The core Bidirectional GRU layers
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # ReLU activation as specified in the paper
        self.relu = nn.ReLU()
        
        # Fully Connected output layer
        # Multiply hidden_dim by 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Pass sequence through the BiGRU
        gru_out, _ = self.bigru(x)
        
        # Extract the output from the final time step
        # gru_out shape: (batch_size, seq_len, hidden_dim * 2)
        final_step_out = gru_out[:, -1, :] 
        
        # Apply activation and output layer
        activated = self.relu(final_step_out)
        prediction = self.fc(activated)
        
        return prediction

def physics_informed_loss(y_pred, y_true, y_phy, lambda_weight):
    """
    Calculates the dual-driven loss balancing data-fitting and physical consistency.
    
    Loss = λ * MSE(Data) + (1 - λ) * MSE(Physics)
    """
    mse_loss = nn.MSELoss()
    
    data_loss = mse_loss(y_pred, y_true)
    physics_loss = mse_loss(y_pred, y_phy)
    
    total_loss = (lambda_weight * data_loss) + ((1.0 - lambda_weight) * physics_loss)
    
    return total_loss