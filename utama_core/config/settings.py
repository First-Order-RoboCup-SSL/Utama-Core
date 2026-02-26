from pathlib import Path

CONTROL_FREQUENCY = 60
TIMESTEP = 1 / CONTROL_FREQUENCY  # interval between frames
SENDING_DELAY = 0  # interval between sending commands to the robot (ms)

BLACKBOARD_NAMESPACE_MAP = {True: "Opp", False: "My"}

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

### SIMULATION SETTINGS ###
ADD_Y_COORD = -3.15
REMOVAL_Y_COORD = -10
TELEPORT_X_COORDS = [0.4, 0.8, 1.2, 1.6, 2, 2.4]

### RSIM Dribbler Settings ###
MIN_RELEASE_SPEED = 0.1  # m/s
RELEASE_GAIN = 0.1
MAX_BALL_SPEED = 3.0  # m/s

### REAL CONTROLLER SETTINGS ###
BAUD_RATE = 115200
PORT = "/dev/ttyUSB0"
KICKER_COOLDOWN_TIME = 10  # in seconds to prevent kicker from being actuated too frequently
KICKER_COOLDOWN_TIMESTEPS = int(KICKER_COOLDOWN_TIME * CONTROL_FREQUENCY)  # in timesteps
KICKER_PERSIST_TIMESTEPS = 10  # in timesteps to persist the kick command

MAX_GAME_HISTORY = 20  # number of previous game states to keep in Game

REPLAY_BASE_PATH = Path.cwd() / "replays"
RENDER_BASE_PATH = Path.cwd() / "renders"

FPS_PRINT_INTERVAL = 1.0  # seconds

### Refiners ###
BALL_MERGE_THRESHOLD = 0.05  # CameraCombiner: distance threshold to merge balls (m)
