import numpy as np

def calculate_vibration(cutting_speed, material_hardness, temperature):
    """
    vibrations based on cutting parameters and temperature. Ideally machines try to achieve minimum vibration
    reasin to calculate: Excessive vibration can lead to poor surface finish, reduced tool life,
      and even machine failure. By modeling vibration, we can predict and mitigate these issues
    """
    base_vibration = (
        0.1 * cutting_speed *    # scaling factor that adjusts the vibration magnitude
        (material_hardness / 100) * 
        (temperature / 50)
    )
    return base_vibration

def add_vibration_noise(base_vibration, noise_level=0.05):
    """
    adding some realistic noise to vibration data
    """
    return base_vibration + np.random.normal(0, noise_level, len(base_vibration))