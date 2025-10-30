# utama_core/config/formulas.py
def max_acceleration(gravity, robot_radius, height_com) -> float:
    return gravity * robot_radius / height_com
