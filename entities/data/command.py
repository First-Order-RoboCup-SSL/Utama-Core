from collections import namedtuple

RobotSimCommand = namedtuple(
    "RobotSimCommand",
    ["local_forward_vel", "local_left_vel", "angular_vel", "kick_vel", "dribbler_vel"],
)
RobotInfo = namedtuple(
    "RobotInfo",
    ["has_ball"],
)