import numpy as np
import pandas as pd
from ..models.pinn import CNCPINN
from ..models.vibration_model import calculate_vibration, add_vibration_noise
from ..models.wear_model import calculate_tool_wear



class CNCDataGenerator:
    def __init__(self, base_temperature = 20.0 , cutting_speed = 100.0, feed_rate = 0.2, material_hardness = 150, cooling_efficiency=1.0):
        self.base_temperature = base_temperature    # Degree C
        self.cutting_speed = cutting_speed          # meters / min
        self.feed_rate = feed_rate                  # mm/rev
        self.material_hardness = material_hardness  # HB Brinell Hardness (more more hard => more heat produced; can chabnge value based on material used)
        self.cooling_efficiency = cooling_efficiency  #1.0 = normal cooling, higher = better cooling
        self.pinn_model = None
    
    def train_with_pinn(self, n_samples=1000):
        """Train PINN model with generated data"""
        # Create training data using our generator settings
        x_train, y_train = CNCPINN.prepare_training_data(
            self.__class__,  # Pass generator class
            n_samples
        )
        # Initialize and train PINN model
        self.pinn_model = CNCPINN()
        CNCPINN.train_pinn(self.pinn_model, x_train, y_train)

    def calculate_temperature(self, time_points):
        """
        calculating temperature using newton law of heating T(t)=T0 + ΔT⋅(1 - e^(-kt))
        if seen on grpah, first temperature will rise quickly then approach stability. (more like a logarithmic curve)
        """
        temperature_rise = (
            4.0 * self.cutting_speed * 
            self.feed_rate * 
            (self.material_hardness / 100)/self.cooling_efficiency
        )

        temperature = self.base_temperature + temperature_rise * (
            1 - np.exp(-0.005 * time_points)  # will make heating curve
        )

        # generating random temperature spikes simulate intensive cutting operations
        num_cycles = len(time_points) // 100
        cycle_points = np.random.choice(len(time_points), num_cycles) #timeswhen the temperature spikes start
        cycle_intensity = np.random.uniform(2.0, 8.0, num_cycles)  # array with intensity for each cycle
        process_variation = np.zeros_like(time_points)             #will store temperature variation due to spikes
        for point, intensity in zip(cycle_points, cycle_intensity):
            # Create a short temperature spike
            window = 50  # duration of each cutting operation
            spike = intensity * np.exp(-0.1 * np.abs(np.arange(-window, window)))
            start_idx = max(0, point - window)
            end_idx = min(len(time_points), point + window)
            process_variation[start_idx:end_idx] += spike[:end_idx-start_idx]


        noise = np.random.normal(0, 0.5, len(time_points)) # adding noise for more realistic data generation

        return temperature + process_variation + noise
        


    # In data_generator.py
def generate_full_dataset(self, duration=1000, sample_rate=1):
        """
        generating a complete dataset including all machine parameters and sensor readings to simulate real machine
        """
        time_points = np.linspace(0, duration, int(duration * sample_rate))
        
        # primary measurements
        temperature = self.calculate_temperature(time_points)
        
        # vibration based on temperature
        vibration = calculate_vibration(
            self.cutting_speed,
            self.material_hardness,
            temperature
        )
        vibration = add_vibration_noise(vibration)
        
        #  tool wear progression
        tool_wear = calculate_tool_wear(
            self.cutting_speed,
            self.material_hardness,
            temperature,
            time_points
        )
        
        # making a comprehensive DataFrame
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-15', 
                                    periods=len(time_points), 
                                    freq=f'{1000/sample_rate}ms'),
            'temperature': temperature,
            'vibration': vibration,
            'tool_wear': tool_wear,
            'cutting_speed': self.cutting_speed,
            'feed_rate': self.feed_rate,
            'material_hardness': self.material_hardness,
            'cooling_efficiency': self.cooling_efficiency
        })