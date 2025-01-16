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