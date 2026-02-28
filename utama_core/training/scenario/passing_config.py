"""Configuration dataclasses for SSL passing drill scenarios."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum


class MacroAction(IntEnum):
    """Parameterized macro-actions for the 3D action space."""

    GO_TO_BALL = 0
    KICK_TO = 1
    DRIBBLE_TO = 2
    MOVE_TO = 3


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
    active_zone_half_width: float = 3.0  # covers receiver spawn at y=-2.5
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
    dribble_force: float = 5.0  # stronger attract for reliable ball holding during reorientation
    dribble_dist_threshold: float = 0.12
    collision_force: float = 10.0  # reduced: VelocityHolonomic makes robot an immovable piston
    max_ball_speed: float = 6.5  # m/s — SSL regulation cap

    # Legacy 6D action space parameters (only used when use_macro_actions=False, use_unified_actions=False)
    action_delta_range: float = 2.0  # max relative position offset in meters
    turn_on_spot_radius_modifier: float = 1.35  # perpendicular velocity scale for ball pivot

    # Action space mode (at most one should be True; both False = legacy 6D)
    use_macro_actions: bool = False  # 3D macro-actions [selector, target_x, target_y]
    use_unified_actions: bool = True  # 3D unified [target_x, target_y, kick_intent]

    # Macro-action configuration (only when use_macro_actions=True)
    n_macro_actions: int = 4  # GO_TO_BALL, KICK_TO, DRIBBLE_TO, MOVE_TO
    dribble_engage_dist: float = 0.15  # distance to start dribble attract in GO_TO_BALL
    decision_interval: int = 1  # reserved for future temporal abstraction

    # Shared kick parameters
    kick_align_threshold: float = 0.95  # cos(angle) to fire kick
    kick_intent_threshold: float = 0.0  # unified: kick_intent > this to attempt kick


@dataclass
class PassingRewardConfig:
    """Reward weights for ASPAC passing drills.

    Dense shaping uses delta-based distances (prev - curr) so standing still = 0.
    """

    # Sparse
    successful_pass: float = 5.0
    ball_out_of_bounds: float = -0.5

    # Dense shaping — passer (delta-based: positive when distance decreases)
    passer_to_ball_weight: float = 1.0  # Phase 1: passer approaches ball
    ball_to_receiver_weight: float = 0.2  # Phase 2: ball→receiver (dribble/pass)

    # Dense shaping — passer orientation
    passer_face_receiver_weight: float = 0.3  # Reward for passer turning to face receiver

    # Ball possession — encourages dribble activation
    has_ball_reward: float = 0.05  # Per-step reward when passer holds ball

    # Kick alignment — one-shot bonus for kicking toward receiver
    kick_alignment_weight: float = 1.0  # cos(facing, to_receiver), clamped >= 0

    # Dense shaping — receiver
    receiver_to_ball_weight: float = 0.1  # Dense shaping: receiver moves toward ball
    receiver_face_ball_weight: float = 0.1  # Reward for facing the ball

    # Active zone enforcement
    out_of_zone_penalty: float = -0.01  # Per-step penalty outside active zone

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

    def __post_init__(self):
        if self.dynamics.use_macro_actions and self.dynamics.use_unified_actions:
            raise ValueError(
                "Cannot enable both use_macro_actions and use_unified_actions. "
                "Set exactly one to True, or both to False for legacy 6D."
            )
