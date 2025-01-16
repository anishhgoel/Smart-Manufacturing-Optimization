import numpy as np
import pandas as pd

class CNCDataGenerator:
    def __init__(self, base_temperature = 20.0 , cutting_speed = 100.0, feed_rate = 0.2, material_hardness = 150):
        self.base_temperature = base_temperature    # Degree C
        self.cutting_speed = cutting_speed          # meters / min
        self.feed_rate = feed_rate                  # mm/rev
        self.material_hardness = material_hardness  # HB Brinell Hardness (more more hard => more heat produced; can chabnge value based on material used)
    
    def calculate_temperature(self, time_points):
        """
        calculating temperature using newton law of heating T(t)=T0 + ΔT⋅(1 - e^(-kt))
        if seen on grpah, first temperature will rise quickly then approach stability. (more like a logarithmic curve)
        """
        temperature_rise = (
            0.1 * self.cutting_speed * 
            self.feed_rate * 
            (self.material_hardness / 100)
        )

        temperature = self.base_temperature + temperature_rise * (
            1 - np.exp(-0.01 * time_points)  # will make heating curve
        )

        noise = np.random.normal(0, 0.5, len(time_points)) # adding noise for more realistic data generation
        return temperature + noise
        