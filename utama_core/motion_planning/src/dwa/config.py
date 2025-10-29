from utama_core.config.settings import ROBOT_RADIUS

SIMULATED_TIMESTEP = 0.05  # seconds
MAX_ACCELERATION = 2  # Measured in ms^2
MAX_SAFETY_RADIUS = ROBOT_RADIUS * 2.5  # m - minimum distance to obstacles we try to maintain
SAFETY_PENALTY_DISTANCE_SQ = 0.3
MAX_SPEED_FOR_FULL_BUBBLE = 1  # m/s at which we apply the full safety bubble
