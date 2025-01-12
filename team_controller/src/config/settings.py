# interval between frames
TIMESTEP = 0.0167

# maximum robot settings
MAX_VEL = 1.5
MAX_ANGULAR_VEL = 8

# sim kick speed
KICK_SPD = 5

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
        "max": MAX_VEL,
        "min": -MAX_VEL,
    },
}

# Simulation controller
ADD_Y_COORD = -3.15
REMOVAL_Y_COORD = -3.8
TELEPORT_X_COORDS = [0.4, 0.8, 1.2, 1.6, 2, 2.4]

# real controller
BAUD_RATE = 115200
PORT = "COM3"
TIMEOUT = 0.1
# s: signed, u: unsigned
SERIAL_BIT_SIZES = {
    "out": {
        "angular_vel": (7, "s"),
        "local_forward_vel": (7, "s"),
        "local_left_vel": (7, "s"),
        "kicker_bottom": (1, "u"),
        "kicker_top": (1, "u"),
        "dribbler": (1, "u"),
    },
    "in": {"kicker_charged": (1, "u"), "has_ball": (1, "u")},
}
ENDIAN = "big"
