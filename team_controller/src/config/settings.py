# Network
LOCAL_HOST = "localhost"
MULTICAST_GROUP = "224.5.23.2"
MULTICAST_GROUP_REFEREE = "224.5.23.1"

VISION_PORT = 10006
YELLOW_TEAM_SIM_PORT = 10302
BLUE_TEAM_SIM_PORT = 10301
REFEREE_PORT = 10003
SIM_COMTROL_PORT = 10300  # IP '127.0.0.1'

# General settings
NUM_ROBOTS = 6

# PID parameters
PID_PARAMS = {
    "oren": {"Kp": 5, "Kd": 0.01, "Ki": 0, "dt": 0.0167, "max": 8, "min": -8},
    "trans": {"Kp": 1.5, "Kd": 0.01, "Ki": 0, "dt": 0.0167, "max": 1.5, "min": -1.5},
}

# Simulation controller
FIELD_Y_COORD = -3150
REMOVAL_Y_COORD = -3800
TELEPORT_X_COORDS = [400, 800, 1200, 1600, 2000, 2400]
