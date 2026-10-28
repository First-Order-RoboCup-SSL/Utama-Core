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
    # Acceleration along the robot's sideways axis, where it is weaker than forward; None is
    # MAX_ACCELERATION. Only `trajsample` plans with it (`Trajectory2D.compute`).
    MAX_SIDEWAYS_ACCELERATION: float | None = None
    # The fastest the robot drives (m/s) and turns (rad/s) with the ball and keeps it: `move()`
    # scales a dribbling command down to both while the robot has the ball. None: no limit.
    CARRY_SPEED: float | None = None
    CARRY_TURN: float | None = None


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
    # The same calibration: sideways 1.6-1.9 m/s^2 (2 steps), forward 4.5-5.3 at 1 m/s.
    MAX_SIDEWAYS_ACCELERATION=1.8,
    # Field runs 2026-10-28: holding the ball at <= 0.1 m/s and <= 1 rad/s (pass_receive,
    # passing_rally) kept it; 0.5 m/s with 2 rad/s, reversing or sideways, lost it in 0.2 s.
    CARRY_SPEED=0.3,
    CARRY_TURN=1.0,
)
