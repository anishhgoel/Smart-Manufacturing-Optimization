import numpy as np

def calculate_tool_wear(cutting_speed, material_hardness, temperature, time_points):
    """
    wear and tear over time based on cutting parameters
    """
    wear_rate = (
        0.00002 * cutting_speed *       #scaling factor that can be adjustable to how durable machine is 
        (material_hardness / 100) * 
        (temperature / 50)
    )
    return np.cumsum(wear_rate * np.ones_like(time_points))  # taking cumulative sum as new wear tear is add on to previous wear and tear
                                                              # assuming, there is no change or fixing or changing parts during the time points