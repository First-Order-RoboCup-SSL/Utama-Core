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
    "oren": {"Kp": 4.5, "Kd": 0, "Ki": 0.3, "dt": 0.0167, "max": 8, "min": -8},
    "trans": {"Kp": 4.5, "Kd": 0, "Ki": 0.2, "dt": 0.0167, "max": 1.5, "min": -1.5},
}

# Simulation controller
ADD_Y_COORD = -3.15
REMOVAL_Y_COORD = -3.8
TELEPORT_X_COORDS = [0.4, 0.8, 1.2, 1.6, 2, 2.4]

# real controller
BAUD_RATE = 115200
PORT = "COM3"
TIMEOUT = 0.1
