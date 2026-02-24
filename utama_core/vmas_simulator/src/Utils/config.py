from dataclasses import dataclass, field


@dataclass(frozen=True)
class SSLFieldConfig:
    """SSL Division B field geometry. All values in meters."""

    half_length: float = 4.5
    half_width: float = 3.0
    goal_width: float = 1.0
    goal_depth: float = 0.18
    defense_area_length: float = 1.0
    defense_area_width: float = 2.0
    robot_radius: float = 0.09  # Matches physical_constants.ROBOT_RADIUS
    ball_radius: float = 0.02135
    boundary_margin: float = 0.3


@dataclass(frozen=True)
class SSLDynamicsConfig:
    """Physics parameters for the VMAS simulation."""

    dt: float = 1.0 / 60.0  # Matches TIMESTEP from settings.py
    substeps: int = 5
    robot_max_speed: float = 2.0  # Matches GRSIM_PARAMS.MAX_VEL
    robot_max_angular_vel: float = 4.0  # Matches GRSIM_PARAMS.MAX_ANGULAR_VEL
    robot_mass: float = 2.5
    ball_mass: float = 0.046
    ball_friction: float = 0.5
    robot_friction: float = 0.1
    kick_impulse: float = 5.0  # Matches GRSIM_PARAMS.KICK_SPD
    dribble_force: float = 1.0
    dribble_dist_threshold: float = 0.12
    collision_force: float = 500.0


@dataclass(frozen=True)
class SSLRewardConfig:
    """Weights for the modular reward system."""

    # Sparse rewards
    goal_scored: float = 100.0
    goal_conceded: float = -100.0

    # Dense shaping rewards
    ball_to_goal_weight: float = 1.0
    agent_to_ball_weight: float = 0.5
    ball_vel_to_goal_weight: float = 0.2

    # Penalties
    out_of_bounds_penalty: float = -1.0

    # Team reward sharing
    team_reward: bool = True


@dataclass(frozen=True)
class SSLScenarioConfig:
    """Top-level scenario configuration combining all sub-configs."""

    n_blue: int = 6
    n_yellow: int = 6
    field_config: SSLFieldConfig = field(default_factory=SSLFieldConfig)
    dynamics: SSLDynamicsConfig = field(default_factory=SSLDynamicsConfig)
    rewards: SSLRewardConfig = field(default_factory=SSLRewardConfig)
    max_steps: int = 3000  # ~50 seconds at 60Hz
    observe_teammates: bool = True
    observe_opponents: bool = True
    observe_ball_velocity: bool = True
