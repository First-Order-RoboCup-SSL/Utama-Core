from dataclasses import dataclass
from numpy.random import normal
from utama_core.global_utils.math_utils import normalise_heading_deg
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise

@dataclass()
class Robot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    z: float = None
    theta: float = None  # degrees
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
    
    
    def add_gaussian_noise(self, noise: RsimGaussianNoise):
        bias = 0
        
        if noise.x_stddev:
            self.x += normal(loc=bias, scale= noise.x_stddev)
            
        if noise.y_stddev:
            self.y += normal(loc=bias, scale= noise.y_stddev)
            
        if noise.th_stddev_deg:
            self.theta = normalise_heading_deg(self.theta + normal(loc=bias, scale=noise.th_stddev_deg))
        
