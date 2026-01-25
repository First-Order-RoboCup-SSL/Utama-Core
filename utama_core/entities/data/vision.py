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
    
    def add_gaussian_noise(
        self,
        x_stddev: float=0.01,
        y_stddev: float=0.01,
        th_stddev_deg: float=0.0,
        bias: float=0.0
    ):        
        self.x += normal(loc=bias, scale=x_stddev)
        self.y += normal(loc=bias, scale=y_stddev)


@dataclass
class VisionData:
    ts: float
    yellow_robots: List[VisionRobotData]
    blue_robots: List[VisionRobotData]
    balls: List[VisionBallData]
