from dataclasses import dataclass
from typing import List
from numpy.random import normal

# position data: meters
# orientation: radians


@dataclass
class VisionBallData:
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class VisionRobotData:
    id: int
    x: float
    y: float
    orientation: float
    
    def add_gaussian_noise(self, sd_in_cm: float=10.0, bias: float=0.0):
        sd_in_m = sd_in_cm/100
        
        self.x += normal(loc=bias, scale=sd_in_m)
        self.y += normal(loc=bias, scale=sd_in_m)


@dataclass
class VisionData:
    ts: float
    yellow_robots: List[VisionRobotData]
    blue_robots: List[VisionRobotData]
    balls: List[VisionBallData]
