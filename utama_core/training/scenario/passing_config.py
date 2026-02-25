"""Configuration dataclasses for SSL passing drill scenarios."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field


@dataclass
class PassingFieldConfig:
    """Full SSL Division B field geometry + active zone for passing drills."""

    # Full field (matches SSLFieldConfig for sim-to-real)
    half_length: float = 4.5
    half_width: float = 3.0
    goal_width: float = 1.0
    goal_depth: float = 0.18
    robot_radius: float = 0.09
    ball_radius: float = 0.02135
    boundary_margin: float = 0.3

    # Active zone — highlighted during rendering, soft constraint via reward
    active_zone_half_length: float = 3.0
    active_zone_half_width: float = 2.0
    active_zone_center_x: float = 0.0
    active_zone_center_y: float = 0.0


@dataclass
class PassingDynamicsConfig:
    """Physics parameters for passing drill agents."""

    dt: float = 1.0 / 60.0
    substeps: int = 5
    robot_max_speed: float = 2.0
    robot_max_angular_vel: float = 4.0
    robot_mass: float = 2.5
    ball_mass: float = 0.046
    ball_friction: float = 0.5
    robot_friction: float = 0.1
    kick_speed: float = 3.0  # m/s — matches RSIM_PARAMS.KICK_SPD (velocity-set, not impulse)
    dribble_force: float = 1.0
    dribble_dist_threshold: float = 0.12
    collision_force: float = 10.0  # reduced: VelocityHolonomic makes robot an immovable piston
    max_ball_speed: float = 6.5  # m/s — SSL regulation cap


@dataclass
class PassingRewardConfig:
    """Reward weights for ASPAC passing drills.

    Dense shaping uses delta-based distances (prev - curr) so standing still = 0.
    """

    # Sparse
    successful_pass: float = 50.0
    ball_out_of_bounds: float = -5.0

    # Dense shaping — passer (delta-based: positive when distance decreases)
    passer_to_ball_weight: float = 10.0  # Phase 1: passer approaches ball
    ball_to_receiver_weight: float = 2.0  # Phase 2: ball→receiver (dribble/pass)

    # Dense shaping — receiver
    receiver_to_ball_weight: float = 0.0  # Disabled: receiver should not chase ball

    # Active zone enforcement
    out_of_zone_penalty: float = -0.1  # Per-step penalty outside active zone

    # ASPAC theoretical parameters (stubs — zero disables)
    envy_free_weight: float = 0.0  # Envy-Free Policy Teaching bonus
    deception_scale: float = 0.0  # Imitative Follower Deception
    displacement_weight: float = 0.0  # Spatial Displacement Error penalty


@dataclass
class PassingScenarioConfig:
    """Top-level config for a passing drill scenario."""

    n_attackers: int = 2
    n_defenders: int = 0
    max_steps: int = 300  # ~5 seconds at 60Hz
    defender_behavior: str = "fixed"  # "fixed" or "active"

    field: PassingFieldConfig = dataclass_field(default_factory=PassingFieldConfig)
    dynamics: PassingDynamicsConfig = dataclass_field(default_factory=PassingDynamicsConfig)
    rewards: PassingRewardConfig = dataclass_field(default_factory=PassingRewardConfig)
