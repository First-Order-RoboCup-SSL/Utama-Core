"""Configuration dataclasses for SSL passing drill scenarios."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum
from typing import Literal


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
class PassingResetRandomizationConfig:
    """Bounded reset distributions for passing drills."""

    benchmark_reset: bool = False

    ball_x_range: tuple[float, float] = (0.75, 1.25)
    ball_y_range: tuple[float, float] = (1.4, 2.3)

    passer_radius_range: tuple[float, float] = (0.85, 1.15)
    passer_angle_range: tuple[float, float] = (0.0, 0.8)

    receiver_x_range: tuple[float, float] = (0.8, 2.1)
    receiver_y_range: tuple[float, float] = (-2.6, -1.8)
    receiver_rot_range: tuple[float, float] = (-0.35, 0.35)

    defender0_distance_range: tuple[float, float] = (0.6, 0.95)
    defender1_x_range: tuple[float, float] = (0.8, 2.6)
    defender1_y_range: tuple[float, float] = (-2.3, -1.7)


@dataclass
class PassingDynamicsConfig:
    """Physics parameters for passing drill agents."""

    dt: float = 1.0 / 60.0
    substeps: int = 5
    robot_max_speed: float = 2.0
    robot_max_angular_vel: float = 4.0
    robot_max_acceleration: float = 4.0
    robot_max_angular_acceleration: float = 50.0
    robot_mass: float = 2.4
    ball_mass: float = 0.043
    ball_friction: float = 0.49
    robot_friction: float = 0.1
    kick_speed: float = 3.0  # m/s — matches RSIM_PARAMS.KICK_SPD (velocity-set, not impulse)
    dribble_force: float = 7.5
    dribble_release_damping: float = 0.35
    dribble_dist_threshold: float = 0.12
    possession_cone_cos: float = 0.5
    pass_min_ball_travel: float = 0.45
    pass_confirm_frames: int = 2
    holder_switch_cooldown_steps: int = 2
    kick_cooldown_steps: int = 10
    collision_force: float = 5.0
    max_ball_speed: float = 6.5  # m/s — SSL regulation cap

    # Legacy 6D action space parameters (only used when use_macro_actions=False, use_unified_actions=False)
    action_delta_range: float = 2.0  # max relative position offset in meters
    turn_on_spot_radius_modifier: float = 1.35  # perpendicular velocity scale for ball pivot

    # Action space mode (at most one should be True; both False = legacy 6D)
    use_macro_actions: bool = False  # 3D macro-actions [selector, target_x, target_y]
    use_unified_actions: bool = True  # 4D unified [target_x, target_y, target_oren, kick_intent]
    physics_mode: Literal["force_based", "kinematic_legacy"] = "force_based"

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
    successful_pass: float = 3.0
    ball_out_of_bounds: float = -0.3

    # Dense shaping — passer (delta-based: positive when distance decreases)
    passer_to_ball_weight: float = 5.0  # Phase 1: passer approaches ball
    ball_to_receiver_weight: float = 0.5  # Phase 2: ball→receiver (dribble/pass)

    # Dense shaping — passer orientation
    passer_face_receiver_weight: float = 0.5  # Reward for passer turning to face receiver

    # Ball possession — encourages dribble activation
    has_ball_reward: float = 0.1  # Per-step reward when passer holds ball

    # Kick alignment — one-shot bonus for kicking toward receiver
    kick_alignment_weight: float = 0.5  # cos(facing, to_receiver), clamped >= 0

    # Dense shaping — receiver
    receiver_to_ball_weight: float = 0.3  # Dense shaping: receiver moves toward ball
    receiver_face_ball_weight: float = 0.3  # Reward for facing the ball

    # Dense shaping — defenders
    defender_delta_weight: float = 0.5

    # Active zone enforcement
    out_of_zone_penalty: float = -0.01  # Per-step penalty outside active zone

    # Curriculum: anneal dense shaping weights over training
    shaping_anneal_start: int = 0  # Frame to begin annealing (0 = from start)
    shaping_anneal_end: int = 0  # Frame to finish annealing (0 = no annealing)
    shaping_anneal_min: float = 0.1  # Minimum multiplier for dense weights


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
    reset_randomization: PassingResetRandomizationConfig = dataclass_field(
        default_factory=PassingResetRandomizationConfig
    )

    def __post_init__(self):
        if self.dynamics.use_macro_actions and self.dynamics.use_unified_actions:
            raise ValueError(
                "Cannot enable both use_macro_actions and use_unified_actions. "
                "Set exactly one to True, or both to False for legacy 6D."
            )
        if self.dynamics.pass_confirm_frames < 1:
            raise ValueError("pass_confirm_frames must be at least 1")
        if self.dynamics.holder_switch_cooldown_steps < 0:
            raise ValueError("holder_switch_cooldown_steps cannot be negative")
        if self.dynamics.physics_mode not in {"force_based", "kinematic_legacy"}:
            raise ValueError("physics_mode must be 'force_based' or 'kinematic_legacy'")
