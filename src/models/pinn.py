import torch
import torch.nn as nn
import numpy as np


class CNCPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=1):
        """
        PINN for CNC machine temperature prediction
        input_dim: cutting_speed, feed_rate, material_hardness, time
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
        t_vals        = x[:, 3:4]
        
    

        #Constraint1 : temperature should be above 20 else we penalize
        # if temperature is above 20, its fine else we have to penalize
      #### (*) i am making it 20 since thats room temp, but can be changed
        min_temp_violation = torch.relu(20.0 - y_pred)

        grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,  # differentiate wrt ALL inputs
        grad_outputs=torch.ones_like(y_pred),   #need to use when output is tensor not a scaler
        create_graph=True  # for multiplt derivatives
    )[0]    # usinfg a 0 here as the result was a tuple with the gradient, but needed only gradient without tuple, so used [0] to extract the gradient
        dT_dt = grads[:, 3:4]        # rate of change of temp w.r.t. time
        #Constraint2 :  to ensure that the rate of temperature change doesn’t exceed a physically realistic limit based on the machine’s cutting speed, feed rate, and material hardness.
        max_heating_rate = 2.0 * cutting_speed * feed_rate * (material_hardness/100)
        heating_rate_violation = torch.relu(torch.abs(dT_dt) - max_heating_rate)

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
        optimizer.zero_grad()  # resets the gradients of the model parameters to zero at the start of each epoch
        
        # forward pass
        y_pred = model(x_train)
        
        # calculating losses
        data_loss = data_criterion(y_pred, y_train)  #data-driven loss
        physics_loss = model.physics_loss(x_train, y_pred) #physics based loss
        
        # combine losses
        total_loss = data_loss + 0.1 * physics_loss
        
        # backward pass
        total_loss.backward()
        optimizer.step() # to optimize model parameters based on gradients
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Data Loss = {data_loss.item():.4f}, '
                f'Physics Loss = {physics_loss.item():.4f}')



def prepare_training_data(generator_class, n_samples=1000):
    """
    Prepare training data from CNC data generator
    
    Parameters:
        generator_class: The CNCDataGenerator class
        n_samples: Number of samples to generate
    Returns:
        x_train: Tensor of input parameters (cutting_speed, feed_rate, material_hardness, time)
        y_train: Tensor of temperature values
    """
    # choosing ranges of parametes; is customizable
    cutting_speeds = np.linspace(50, 200, 5)   
    feed_rates = np.linspace(0.1, 0.4, 5)
    materials = [75, 150]
    
    # diff times to sample, kinda like 0 -> 100 in steps of 10
    time_array = np.linspace(0, 100, 11)
    
    inputs = []
    outputs = []
    
    for cs in cutting_speeds:
        for fr in feed_rates:
            for mh in materials:
                # instantiating the generator with these params
                gen = generator_class(
                    cutting_speed=cs,
                    feed_rate=fr,
                    material_hardness=mh
                )
                # the temperature curve at times
                temps = gen.calculate_temperature(time_array)
                
                #  storing: (cs, fr, mh, t) -> T for each step
                for t_val, temp_val in zip(time_array, temps):
                    inputs.append([cs, fr, mh, t_val])
                    outputs.append(temp_val)
    
    x_train = torch.tensor(inputs, dtype=torch.float32)
    x_train.requires_grad_(True)
    y_train = torch.tensor(outputs, dtype=torch.float32).reshape(-1, 1)
    return x_train, y_train