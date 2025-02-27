from typing import Tuple, Union, Optional, List, NamedTuple

# position data: meters
# orientation: radians

class BallData(NamedTuple):
    x: float
    y: float
    z: float
    confidence: float

class RobotData(NamedTuple):
    id: int
    x: float
    y: float
    orientation: float

class FrameData(NamedTuple):
    ts: float
    yellow_robots: List[RobotData]
    blue_robots: List[RobotData]
    ball: List[BallData]
