import torch
import torch.nn as nn
import numpy as np


class CNCPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=50, output_dim=1):
        """
        PINN for CNC machine temperature prediction
        input_dim: cutting_speed, feed_rate, material_hardness
        output_dim: temperature
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

    def physics_loss(self, x, y_pred):
        """
        Physics-based constraints:
        1. Temperature should always be above (20°C)
        2. Rate of temperature change has physical limits
        3. Material hardness proportionally affects temperature
        """
        # unpacking; also usning 0:1 instead of 0 as we need 2d tensor instead of 1d tensor
        cutting_speed = x[:, 0:1]
        feed_rate = x[:, 1:2]
        material_hardness = x[:, 2:3]
        
        # constraints -> calculating rate of change here. ; squeeze is to remove unnecessary dimensions
        temp_gradient = torch.gradient(y_pred.squeeze(), dim=0)[0]
        
        # if temperature is above 20, its fine else we have to penalize
      #### (*) i am making it 20 since thats room temp, but can be changed
        min_temp_violation = torch.relu(20.0 - y_pred)