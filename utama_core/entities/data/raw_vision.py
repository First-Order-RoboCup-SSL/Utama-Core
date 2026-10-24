from dataclasses import dataclass
from typing import List, Optional, Tuple

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
class RawVisionData:
    ts: float
    yellow_robots: List[RawRobotData]
    blue_robots: List[RawRobotData]
    balls: List[RawBallData]
    camera_id: int
    # Where that camera is (x, y, height), from the geometry's calibration (ssl-vision, grSim); None when
    # not sent (rsim) or not calibrated yet (height 0).
    camera_position: Optional[Tuple[float, float, float]] = None
