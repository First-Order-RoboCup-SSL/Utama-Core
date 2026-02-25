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
    """Physics parameters for the VMAS simulation, matched to rSim."""

    dt: float = 1.0 / 60.0  # Matches TIMESTEP from settings.py
    substeps: int = 5
    robot_max_speed: float = 2.0  # Matches RSIM_PARAMS.MAX_VEL
    robot_max_angular_vel: float = 4.0  # Matches RSIM_PARAMS.MAX_ANGULAR_VEL
    robot_mass: float = 2.4  # rSim: 2.2kg chassis + 4x0.05kg wheels
    ball_mass: float = 0.043  # rSim ball mass
    ball_friction: float = 0.49  # rSim: mu * g = 0.05 * 9.81
    ball_drag: float = 0.004  # rSim linear_damping
    robot_friction: float = 0.1
    kick_speed: float = 3.0  # Max ball velocity after kick (m/s)
    kick_damp_factor: float = 0.2  # rSim: preserve 20% of existing ball momentum on kick
    kick_cooldown_steps: int = 10  # rSim: 10-iteration cooldown between kicks
    dribble_dist_threshold: float = 0.12  # robot_radius + ball_radius + margin (VMAS uses sphere robots)
    collision_force: float = 5.0  # Low penalty stiffness; actual momentum from our push/bounce
    ball_speed_dead_zone: float = 0.01  # rSim: stop ball below this speed (m/s)
    ball_restitution: float = 0.5  # rSim bounce coefficient
    ball_bounce_vel_threshold: float = 0.1  # rSim: minimum speed for bounce
    robot_max_acceleration: float = 4.0  # Matches RSIM_PARAMS.MAX_ACCELERATION
    robot_max_angular_acceleration: float = 50.0  # Matches RSIM_PARAMS.MAX_ANGULAR_ACCELERATION
    ball_push_transfer_coeff: float = 0.8  # Velocity transfer on robot-ball contact (dribbler off)


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
