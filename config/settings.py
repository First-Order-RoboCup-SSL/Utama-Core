from numpy import pi

# interval between frames
TIMESTEP = 1 / 60
# interval between sending commands to the robot (ms)
SENDING_DELAY = 0

# maximum (real and sim) robot settings
REAL_MAX_VEL = 0.2
# any slower and the robots become unstable
REAL_MAX_ANGULAR_VEL = 0.5

# maximum (real and sim) robot settings
MAX_VEL = 2
# any slower and the robots become unstable
MAX_ANGULAR_VEL = 4

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

MAX_CAMERAS = 10  # Less is fine

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
REMOVAL_Y_COORD = -10
TELEPORT_X_COORDS = [0.4, 0.8, 1.2, 1.6, 2, 2.4]

# real controller
BAUD_RATE = 115200
PORT = "/dev/ttyACM0"
AUTH_STR = "<READY!>"
MAX_INITIALIZATION_TIME = 5
TIMEOUT = 0.1

MAX_GAME_HISTORY = 20
