from typing import List
from dataclasses import dataclass

# position data: meters
# orientation: radians

@dataclass
class BallData:
    x: float
    y: float
    z: float
    confidence: float

@dataclass
class RobotData:
    id: int
    x: float
    y: float
    orientation: float

@dataclass
class FrameData:
    ts: float
    yellow_robots: List[RobotData]
    blue_robots: List[RobotData]
    ball: List[BallData]
