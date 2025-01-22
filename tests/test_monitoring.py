# Create new file: tests/test_monitoring.py
import numpy as np
import matplotlib.pyplot as plt
from src.data.data_generator import CNCDataGenerator
from src.utils.monitoring import CNCMonitor
from src.models.pinn import CNCPINN

def test_full_system():
    # Initialize system components
    generator = CNCDataGenerator(
        cutting_speed=150.0,
        material_hardness=150,  # Steel
        cooling_efficiency=1.0
    )
    
    # Generate data
    data = generator.generate_full_dataset(duration=3600, sample_rate=1)
    
    # Train PINN model
    generator.train_with_pinn(n_samples=1000)
    
    # Monitor system
    monitor = CNCMonitor()
    status = monitor.analyze_data(data)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Temperature plot
    axes[0].plot(data['timestamp'], data['temperature'])
    axes[0].set_title('Temperature Over Time')
    axes[0].grid(True)
    
    # Vibration plot
    axes[1].plot(data['timestamp'], data['vibration'])
    axes[1].set_title('Vibration Over Time')
    axes[1].grid(True)
    
    # Tool wear plot
    axes[2].plot(data['timestamp'], data['tool_wear'])
    axes[2].set_title('Tool Wear Progression')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print system status
    print("\nSystem Status Report:")
    print(f"Machine Health: {'Healthy' if status.is_healthy else 'Warning'}")
    print(f"Temperature: {status.temperature_status}")
    print(f"Vibration: {status.vibration_status}")
    print(f"Tool Wear: {status.tool_wear_status}")
    if status.anomalies:
        print("\nAnomalies Detected:")
        for param, timestamps in status.anomalies.items():
            print(f"{param}: {len(timestamps)} anomalies detected")

if __name__ == "__main__":
    test_full_system()