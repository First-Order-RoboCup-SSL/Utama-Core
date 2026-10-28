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
    # From a command being sent to vision seeing the robot respond (radio, motors and vision
    # together). `move()` plans from where our robots will be by then (`PoseAhead`).
    CONTROL_LATENCY_S: float = 0.0


GRSIM_PARAMS = RobotParams(
    MAX_VEL=2,
    MAX_ANGULAR_VEL=4,
    MAX_ACCELERATION=4,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
)

RSIM_PARAMS = RobotParams(
    MAX_VEL=2,
    MAX_ANGULAR_VEL=4,
    MAX_ACCELERATION=4,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
)

REAL_PARAMS = RobotParams(
    MAX_VEL=1,
    MAX_ANGULAR_VEL=4,
    MAX_ACCELERATION=4,
    MAX_ANGULAR_ACCELERATION=50,
    KICK_SPD=5,
    DRIBBLE_SPD=3,
    CHIP_ANGLE=pi / 4,
    # Median loop delay of yellow 1's calibration, 2026-10-27 (58-112 ms per step):
    # replays/field/20261027_225327_real_calibrate.calibration.json. Re-measure with `pixi run calibrate`.
    CONTROL_LATENCY_S=0.079,
)
