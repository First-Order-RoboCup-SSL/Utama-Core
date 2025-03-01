from typing import List
from dataclasses import dataclass

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


@dataclass
class VisionData:
    ts: float
    yellow_robots: List[VisionRobotData]
    blue_robots: List[VisionRobotData]
    ball: List[VisionBallData]
