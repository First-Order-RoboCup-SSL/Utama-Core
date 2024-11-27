import numpy as np

# Network
LOCAL_HOST = 'localhost'
MULTICAST_GROUP = '224.5.23.2'
MULTICAST_GROUP_REFEREE='224.5.23.1'

VISION_PORT = 10006
YELLOW_TEAM_SIM_PORT = 10302
BLUE_TEAM_SIM_PORT = 10301
REFEREE_PORT = 10003
SIM_CONTROL_PORT = 10300 # IP '127.0.0.1'

# General settings
NUM_ROBOTS = 6

# PID parameters
PID_PARAMS = {
    'oren': {'Kp': 5, 'Kd': 0.01, 'Ki': 0, 'dt': 0.0167, 'max': 8, 'min': -8},
    'trans': {'Kp': 1.5, 'Kd': 0.01, 'Ki': 0, 'dt': 0.0167, 'max': 1.5, 'min': -1.5}
}

# Starting positions for yellow team
YELLOW_START = [
            (4200.0, 0.0, np.pi),
            (3400.0, -200.0, np.pi),
            (3400.0, 200.0, np.pi),
            (700.0, 0.0, np.pi),
            (700.0, 2250.0, np.pi),
            (700.0, -2250.0, np.pi),
            (2000.0, 750.0, np.pi),
            (2000.0, -750.0, np.pi),
            (2000.0, 1500.0, np.pi),
            (2000.0, -1500.0, np.pi),
            (2000.0, 2250.0, np.pi)
        ]

# Simulation controller
FIELD_Y_COORD = -3150
REMOVAL_Y_COORD = -3800
TELEPORT_X_COORDS = [400, 800, 1200, 1600, 2000, 2400]