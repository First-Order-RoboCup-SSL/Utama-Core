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
    
    def add_noise(self):
        self.x += normal(loc=0.0, scale=0.05)
        self.y += normal(loc=0.0, scale=0.05)


@dataclass
class VisionData:
    ts: float
    yellow_robots: List[VisionRobotData]
    blue_robots: List[VisionRobotData]
    balls: List[VisionBallData]
