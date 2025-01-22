# Create new file: src/utils/monitoring.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class MachineStatus:
    """Represents the current status of the CNC machine"""
    is_healthy: bool
    temperature_status: str
    vibration_status: str
    tool_wear_status: str
    estimated_maintenance_time: float
    anomalies: Dict[str, List[str]]

class CNCMonitor:
    def __init__(self, 
                 temp_threshold=160.0,
                 vibration_threshold=2.0,
                 tool_wear_threshold=0.8):
        self.temp_threshold = temp_threshold
        self.vibration_threshold = vibration_threshold
        self.tool_wear_threshold = tool_wear_threshold
    
    def analyze_data(self, data: pd.DataFrame) -> MachineStatus:
        """
        Analyzes machine data for potential issues and maintenance needs.
        Uses statistical analysis and threshold-based rules.
        """
        # Temperature analysis
        temp_status = "Normal"
        if data['temperature'].max() > self.temp_threshold:
            temp_status = "Warning: High Temperature"
        
        # Vibration analysis (using statistical outlier detection)
        vibration_mean = data['vibration'].mean()
        vibration_std = data['vibration'].std()
        vibration_status = "Normal"
        if any(data['vibration'] > vibration_mean + 3*vibration_std):
            vibration_status = "Warning: Excessive Vibration"
        
        # Tool wear prediction
        wear_rate = np.diff(data['tool_wear']).mean()
        current_wear = data['tool_wear'].iloc[-1]
        if wear_rate <= 0:
            # edge case: if wear_rate is 0 or negative (shouldn't happen normally)
            time_to_maintenance_hours = float('inf')  # or 0
        else:
            time_to_maintenance_seconds = (self.tool_wear_threshold - current_wear) / wear_rate
            time_to_maintenance_seconds = max(time_to_maintenance_seconds, 0)
            time_to_maintenance_hours = time_to_maintenance_seconds / 3600.0
        
        # Collect anomalies
        anomalies = {}
        for column in ['temperature', 'vibration', 'tool_wear']:
            window_size = 100
            rolling_mean = data[column].rolling(window=window_size, center=True).mean()
            rolling_std = data[column].rolling(window=window_size, center=True).std()
            rolling_mean = rolling_mean.bfill().ffill()
            rolling_std  = rolling_std.bfill().ffill()
            
            rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
            rolling_std  = rolling_std.fillna(method='bfill').fillna(method='ffill')
            
            z_scores = np.abs((data[column] - rolling_mean) / rolling_std)
            anomaly_timestamps = data.index[z_scores > 3].tolist()
            if anomaly_timestamps:
                anomalies[column] = anomaly_timestamps
        
        return MachineStatus(
            is_healthy=len(anomalies) == 0,
            temperature_status=temp_status,
            vibration_status=vibration_status,
            tool_wear_status=f"Estimated {time_to_maintenance_hours:.1f} hours until maintenance",
            estimated_maintenance_time=time_to_maintenance_hours,
            anomalies=anomalies
        )