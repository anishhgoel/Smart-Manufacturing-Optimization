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
        

        #Constraint1 : temperature should be above 20 else we penalize
        # if temperature is above 20, its fine else we have to penalize
      #### (*) i am making it 20 since thats room temp, but can be changed
        min_temp_violation = torch.relu(20.0 - y_pred)

        #Constraint2 :  to ensure that the rate of temperature change doesn’t exceed a physically realistic limit based on the machine’s cutting speed, feed rate, and material hardness.
        max_heating_rate = 2.0 * cutting_speed * feed_rate * (material_hardness/100)
        heating_rate_violation = torch.relu(torch.abs(temp_gradient) - max_heating_rate)

        physics_loss = (
            torch.mean(min_temp_violation) + 
            torch.mean(heating_rate_violation)
        )
        
        return physics_loss
    
    
    def train_pinn(model, x_train, y_train, epochs=1000):
        """
        training the PINN model with both data and physics losses
        """
        # using adam( daptive Moment Estimation) optimizing algo since it used momentum and has adaptive learning rate. we specify base lr/ adam 
        # determines learning rate and multiplies with base learning rate we provide
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        data_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # forward pass
            y_pred = model(x_train)
            
            # alculation losses
            data_loss = data_criterion(y_pred, y_train)
            physics_loss = model.physics_loss(x_train, y_pred)
            
            # Total loss
            total_loss = data_loss + 0.1 * physics_loss
            
            # Backward pass
            total_loss.backward()
            # making adjustmwnts to weights and biases
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Data Loss = {data_loss.item():.4f}, Physics Loss = {physics_loss.item():.4f}')
