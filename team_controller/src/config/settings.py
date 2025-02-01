from numpy import pi

# interval between frames
TIMESTEP = 0.0167

# maximum (real and sim) robot settings
MAX_VEL = 0.2
# any slower and the robots become unstable
MAX_ANGULAR_VEL = 0.5

ROBOT_RADIUS = 0.09  # TODO: probably not the best place to put this

# sim kick speed
KICK_SPD = 5
DRIBBLE_SPD = 3
CHIP_ANGLE = pi / 4

# Network
LOCAL_HOST = "localhost"
MULTICAST_GROUP = "224.5.23.2"
MULTICAST_GROUP_REFEREE = "224.5.23.1"

VISION_PORT = 10006
YELLOW_TEAM_SIM_PORT = 10302
BLUE_TEAM_SIM_PORT = 10301
REFEREE_PORT = 10003
SIM_CONTROL_PORT = 10300  # IP '127.0.0.1'

# General settings
NUM_ROBOTS = 6

# PID parameters
PID_PARAMS = {
    "oren": {
        "Kp": 4.5,
        "Kd": 0,
        "Ki": 0.3,
        "dt": TIMESTEP,
        "max": MAX_ANGULAR_VEL,
        "min": -MAX_ANGULAR_VEL,
    },
    "trans": {
        "Kp": 4.5,
        "Kd": 0,
        "Ki": 0.2,
        "dt": TIMESTEP,
        "max_vel": MAX_VEL,
    },
}

# Simulation controller
ADD_Y_COORD = -3.15
REMOVAL_Y_COORD = -3.8
TELEPORT_X_COORDS = [0.4, 0.8, 1.2, 1.6, 2, 2.4]

# real controller
BAUD_RATE = 115200
PORT = "/dev/ttyACM1"
AUTH_STR = "<READY>"
MAX_INITIALIZATION_TIME = 5
TIMEOUT = 0.1
# NOTE: angular_vel, local_forward_vel, local_left_vel are 16 bit floating point.
# they should not be changed to any arbitrary value.
SERIAL_BIT_SIZES = {
    "out": {
        "angular_vel": 16,
        "local_forward_vel": 16,
        "local_left_vel": 16,
        "kicker_bottom": 1,
        "kicker_top": 1,
        "dribbler": 1,
        "robot_id": 5,
        "spare": 8,
    },
    "in": {"has_ball": 1},  # TODO: add "kicker_charged": 1,
}
ENDIAN = "big"
