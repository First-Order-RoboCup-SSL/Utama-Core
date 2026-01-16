from dataclasses import dataclass
from numpy.random import normal


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
    
    
    def add_gaussian_noise(self, sd_in_cm: float=10.0, bias: float=0):
        sd_in_m = sd_in_cm / 100
        
        self.x += normal(loc=bias, scale=sd_in_m)
        self.y += normal(loc=bias, scale=sd_in_m)
