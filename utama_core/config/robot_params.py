from dataclasses import dataclass
from math import pi


@dataclass(frozen=True, slots=True)
class RobotParams:
    MAX_VEL: float
    MAX_ANGULAR_VEL: float
    MAX_ACCELERATION: float
    MAX_ANGULAR_ACCELERATION: float
    KICK_SPD: float
    DRIBBLE_SPD: float
    CHIP_ANGLE: float


GRSIM_PARAMS = RobotParams(
    MAX_VEL=2,
    MAX_ANGULAR_VEL=4,
    MAX_ACCELERATION=8,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
)

RSIM_PARAMS = RobotParams(
    MAX_VEL=2,
    MAX_ANGULAR_VEL=4,
    MAX_ACCELERATION=8,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
)

REAL_PARAMS = RobotParams(
    MAX_VEL=0.5,
    MAX_ANGULAR_VEL=1,
    MAX_ACCELERATION=4,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
)
