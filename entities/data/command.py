from collections import namedtuple

RobotSimCommand = namedtuple(
    "RobotSimCommand",
    ["local_forward_vel", "local_left_vel", "angular_vel", "kick_spd", "kick_angle", "dribbler_spd"],
)
RobotInfo = namedtuple(
    "RobotInfo",
    ["has_ball"],
)