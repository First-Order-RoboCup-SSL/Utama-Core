from dataclasses import dataclass
from numpy.random import normal
from utama_core.global_utils.math_utils import deg_to_rad, normalise_heading

@dataclass()
class Robot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    z: float = None
    theta: float = None
    v_x: float = 0
    v_y: float = 0
    v_theta: float = 0
    kick_v_x: float = 0
    kick_v_z: float = 0
    dribbler: bool = False
    infrared: bool = False
    wheel_speed: bool = False
    v_wheel0: float = 0  # rad/s
    v_wheel1: float = 0  # rad/s
    v_wheel2: float = 0  # rad/s
    v_wheel3: float = 0  # rad/s
    
    
    def add_gaussian_noise(self, x_sd_cm, y_sd_cm, th_sd_deg):
        bias = 0
        
        self.x += normal(loc=bias, scale= 0 / 100)
        self.y += normal(loc=bias, scale= 0 / 100)
        self.theta = normalise_heading(self.theta + normal(loc=bias, scale= deg_to_rad(0)))
