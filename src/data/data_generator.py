import numpy as np
import pandas as pd

class CNCDataGenerator:
    def __init__(self, base_temperature = 20.0 , cutting_speed = 100.0, feed_rate = 0.2, material_hardness = 150, cooling_efficiency=1.0):
        self.base_temperature = base_temperature    # Degree C
        self.cutting_speed = cutting_speed          # meters / min
        self.feed_rate = feed_rate                  # mm/rev
        self.material_hardness = material_hardness  # HB Brinell Hardness (more more hard => more heat produced; can chabnge value based on material used)
        self.cooling_efficiency = cooling_efficiency  #1.0 = normal cooling, higher = better cooling
    
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
        cycle_points = np.random.choice(len(time_points), num_cycles)
        cycle_intensity = np.random.uniform(2.0, 8.0, num_cycles)
        process_variation = np.zeros_like(time_points)
        for point, intensity in zip(cycle_points, cycle_intensity):
            # Create a short temperature spike
            window = 50  # Duration of each cutting operation
            spike = intensity * np.exp(-0.1 * np.abs(np.arange(-window, window)))
            start_idx = max(0, point - window)
            end_idx = min(len(time_points), point + window)
            process_variation[start_idx:end_idx] += spike[:end_idx-start_idx]


        noise = np.random.normal(0, 0.5, len(time_points)) # adding noise for more realistic data generation

        return temperature + process_variation + noise
        