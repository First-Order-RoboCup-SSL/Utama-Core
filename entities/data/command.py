from typing import NamedTuple

# velocities: meters per second
# angular_vel: radians per second
class RobotCommand(NamedTuple):
    local_forward_vel: float
    local_left_vel: float
    angular_vel: float
    kick: bool
    chip: bool
    dribble: bool

# Grsim Inverse Kinematics
class RobotVelCommand(NamedTuple):
    front_right: float
    back_right: float
    front_left: float
    back_left: float
    kick: bool
    chip: bool
    dribble: bool

class RobotResponse(NamedTuple):
    has_ball: bool
# TODO: add kicker charged to robot info

