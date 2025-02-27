from dataclasses import dataclass
from typing import List

# Unit: m

@dataclass
class RawRobotData:
    id: int
    x: float
    y: float
    orientation: float
    confidence: float

@dataclass
class RawBallData:
    x: float
    y: float
    z: float
    confidence: float

@dataclass
class RawFrameData:
    ts: float
    yellow_robots: List[RawRobotData]
    blue_robots: List[RawRobotData]
    balls: List[RawBallData]
    camera_id: int
