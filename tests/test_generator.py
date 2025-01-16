# test_generator.py
from src.data.data_generator import CNCDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Test different materials
aluminum = CNCDataGenerator(
        base_temperature=20.0,
        cutting_speed=200.0,     
        feed_rate=0.3,            
        material_hardness=75,    
        cooling_efficiency=1.2   
    )

steel = CNCDataGenerator(
    base_temperature=20.0,
    cutting_speed=150.0,      
    feed_rate=0.2,
    material_hardness=150,    
    cooling_efficiency=1.0    
)

# time points
time_points = np.linspace(0, 10000, 1000)

# temperature data
temp_aluminum = aluminum.calculate_temperature(time_points)
temp_steel = steel.calculate_temperature(time_points)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_points, temp_aluminum, label='Aluminum (75 HB)', color='blue')
plt.plot(time_points, temp_steel, label='Steel (150 HB)', color='red')
plt.title('CNC Machine Temperature Over Time - Material Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.show()