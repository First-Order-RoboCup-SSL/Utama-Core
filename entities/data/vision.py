from typing import Tuple, Union, Optional, List, NamedTuple

# position data: meters
# orientation: radians

class BallData(NamedTuple):
    x: float
    y: float
    z: float

class RobotData(NamedTuple):
    x: float
    y: float
    orientation: float

class FrameData(NamedTuple):
    ts: float
    yellow_robots: List[RobotData]
    blue_robots: List[RobotData]
    ball: List[BallData]

class TeamRobotCoords(NamedTuple):
    team_color: str
    robots: List[RobotData]
