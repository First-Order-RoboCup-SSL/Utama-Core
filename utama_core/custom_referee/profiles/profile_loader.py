"""Profile loader: parses YAML referee profiles into typed dataclasses."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Rule config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GoalDetectionConfig:
    enabled: bool = True
    cooldown_seconds: float = 1.0


@dataclass
class OutOfBoundsConfig:
    enabled: bool = True
    free_kick_assigner: str = "last_touch"


@dataclass
class DefenseAreaConfig:
    enabled: bool = True
    max_defenders: int = 1
    attacker_infringement: bool = True


@dataclass
class KeepOutConfig:
    enabled: bool = True
    radius_meters: float = 0.5
    violation_persistence_frames: int = 30


@dataclass
class BallSpeedConfig:
    enabled: bool = True
    max_speed_mps: float = 6.5  # SSL Division B kick-speed limit


@dataclass
class DoubleTouchConfig:
    enabled: bool = True


@dataclass
class KeeperHeldBallConfig:
    enabled: bool = True
    # SSL rulebook §8.4.1: 5s (Division A) or 10s (Division B). No
    # division-selection concept exists elsewhere in this profile loader, so
    # default to the (looser) Division B value; override per-profile for A.
    max_hold_seconds: float = 10.0


@dataclass
class ExcessiveDribblingConfig:
    enabled: bool = True
    max_dribble_meters: float = 1.0


@dataclass
class RobotStopSpeedConfig:
    enabled: bool = True
    max_speed_mps: float = 1.5
    grace_seconds: float = 2.0


@dataclass
class PushingConfig:
    """SSL rulebook §8.4.1 "Pushing" — see PushingRule's docstring."""

    enabled: bool = True
    min_closing_speed_mps: float = 0.05
    similar_force_margin_mps: float = 0.15
    persistence_frames: int = 15


@dataclass
class CrashingConfig:
    """SSL rulebook §8.4.2 "Crashing" — see CrashingRule's docstring."""

    enabled: bool = True
    fault_speed_threshold_mps: float = 1.5  # SSL rulebook value
    both_fault_threshold_mps: float = 0.3  # SSL rulebook value
    retrigger_cooldown_seconds: float = 2.0  # SSL rulebook value


@dataclass
class DefenseAreaStoppageConfig:
    """SSL rulebook §8.4.1 "Robot Too Close To Opponent Defense Area" (the
    stoppage-time version) — see DefenseAreaStoppageRule's docstring."""

    enabled: bool = True
    min_distance_meters: float = 0.2
    grace_seconds: float = 2.0


@dataclass
class BallPlacementInterferenceConfig:
    """SSL rulebook §8.4.3 "Ball Placement Interference" — see
    BallPlacementInterferenceRule's docstring."""

    enabled: bool = True
    stadium_radius_meters: float = 0.5
    grace_seconds: float = 2.0


@dataclass
class RulesConfig:
    goal_detection: GoalDetectionConfig = field(default_factory=GoalDetectionConfig)
    out_of_bounds: OutOfBoundsConfig = field(default_factory=OutOfBoundsConfig)
    defense_area: DefenseAreaConfig = field(default_factory=DefenseAreaConfig)
    keep_out: KeepOutConfig = field(default_factory=KeepOutConfig)
    ball_speed: BallSpeedConfig = field(default_factory=BallSpeedConfig)
    double_touch: DoubleTouchConfig = field(default_factory=DoubleTouchConfig)
    keeper_held_ball: KeeperHeldBallConfig = field(default_factory=KeeperHeldBallConfig)
    excessive_dribbling: ExcessiveDribblingConfig = field(default_factory=ExcessiveDribblingConfig)
    robot_stop_speed: RobotStopSpeedConfig = field(default_factory=RobotStopSpeedConfig)
    pushing: PushingConfig = field(default_factory=PushingConfig)
    crashing: CrashingConfig = field(default_factory=CrashingConfig)
    defense_area_stoppage: DefenseAreaStoppageConfig = field(default_factory=DefenseAreaStoppageConfig)
    ball_placement_interference: BallPlacementInterferenceConfig = field(
        default_factory=BallPlacementInterferenceConfig
    )


# ---------------------------------------------------------------------------
# Game config
# ---------------------------------------------------------------------------


@dataclass
class AutoAdvanceConfig:
    """Controls which state-machine transitions fire automatically.

    Set all to False for physical environments where a human operator must
    explicitly advance the state to prevent robots from moving unexpectedly.
    """

    # STOP → queued restart command when all robots have cleared the ball.
    stop_to_next_command: bool = True
    # PREPARE_KICKOFF_* → NORMAL_START after prepare_duration_seconds when
    # the kicker is inside the centre circle.
    prepare_kickoff_to_normal: bool = True
    # PREPARE_PENALTY_* → NORMAL_START after prepare_duration_seconds when
    # the kicker reaches the penalty mark.
    prepare_penalty_to_normal: bool = True
    # DIRECT_FREE_* → NORMAL_START when kicker is in position and defenders
    # have cleared.
    direct_free_to_normal: bool = True
    # BALL_PLACEMENT_* → next_command when ball reaches placement target.
    ball_placement_to_next: bool = True
    # NORMAL_START → FORCE_START after kickoff_timeout_seconds if ball hasn't
    # moved (catches a stuck kickoff).
    normal_start_to_force: bool = True


@dataclass
class GameConfig:
    half_duration_seconds: float = 300.0
    kickoff_team: str = "yellow"
    # If True, skip PREPARE_KICKOFF and issue FORCE_START automatically after
    # stop_duration_seconds.  Optional fast-path for continuous-play scenarios.
    force_start_after_goal: bool = False
    # How long to stay in STOP before auto-advancing (only when
    # force_start_after_goal=True).  Set to 0.0 to advance immediately.
    stop_duration_seconds: float = 3.0
    # How long to stay in PREPARE_KICKOFF_* before auto-issuing NORMAL_START.
    # Gives robots time to reach their kickoff formation.  SSL Div B allows
    # 10 s to execute the kick after NORMAL_START, so this just covers the
    # formation phase.
    prepare_duration_seconds: float = 3.0
    # How long after NORMAL_START (kickoff/free-kick) before FORCE_START is
    # issued automatically if the ball has not moved.  SSL rule: 10 s.
    kickoff_timeout_seconds: float = 10.0
    auto_advance: AutoAdvanceConfig = field(default_factory=AutoAdvanceConfig)


# ---------------------------------------------------------------------------
# Top-level profile
# ---------------------------------------------------------------------------


@dataclass
class RefereeProfile:
    profile_name: str
    rules: RulesConfig
    game: GameConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_profile(name_or_path: str) -> RefereeProfile:
    """Load a RefereeProfile from a built-in name or an absolute/relative path.

    Built-in names: "simulation", "human".
    """
    aliases = {"strict_ai": "simulation", "arcade": "human"}
    if name_or_path in aliases:
        warnings.warn(
            f"Profile '{name_or_path}' is deprecated; use '{aliases[name_or_path]}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name_or_path = aliases[name_or_path]

    p = Path(name_or_path)
    if not p.is_absolute():
        # Try built-in profiles directory
        candidate = _PROFILES_DIR / f"{name_or_path}.yaml"
        if candidate.exists():
            p = candidate
        elif not p.exists():
            raise FileNotFoundError(f"Profile '{name_or_path}' not found as a built-in name or file path.")

    with open(p, "r") as fh:
        data = yaml.safe_load(fh)

    return _parse_profile(data)


def _parse_profile(data: dict) -> RefereeProfile:
    rules_d = data.get("rules", {})

    gd = rules_d.get("goal_detection", {})
    goal_cfg = GoalDetectionConfig(
        enabled=gd.get("enabled", True),
        cooldown_seconds=gd.get("cooldown_seconds", 1.0),
    )

    ob = rules_d.get("out_of_bounds", {})
    oob_cfg = OutOfBoundsConfig(
        enabled=ob.get("enabled", True),
        free_kick_assigner=ob.get("free_kick_assigner", "last_touch"),
    )

    da = rules_d.get("defense_area", {})
    da_cfg = DefenseAreaConfig(
        enabled=da.get("enabled", True),
        max_defenders=da.get("max_defenders", 1),
        attacker_infringement=da.get("attacker_infringement", True),
    )

    ko = rules_d.get("keep_out", {})
    ko_cfg = KeepOutConfig(
        enabled=ko.get("enabled", True),
        radius_meters=ko.get("radius_meters", 0.5),
        violation_persistence_frames=ko.get("violation_persistence_frames", 30),
    )

    bs = rules_d.get("ball_speed", {})
    bs_cfg = BallSpeedConfig(
        enabled=bs.get("enabled", True),
        max_speed_mps=bs.get("max_speed_mps", 6.5),
    )

    dt = rules_d.get("double_touch", {})
    dt_cfg = DoubleTouchConfig(
        enabled=dt.get("enabled", True),
    )

    khb = rules_d.get("keeper_held_ball", {})
    khb_cfg = KeeperHeldBallConfig(
        enabled=khb.get("enabled", True),
        max_hold_seconds=khb.get("max_hold_seconds", 10.0),
    )

    ed = rules_d.get("excessive_dribbling", {})
    ed_cfg = ExcessiveDribblingConfig(
        enabled=ed.get("enabled", True),
        max_dribble_meters=ed.get("max_dribble_meters", 1.0),
    )

    rss = rules_d.get("robot_stop_speed", {})
    rss_cfg = RobotStopSpeedConfig(
        enabled=rss.get("enabled", True),
        max_speed_mps=rss.get("max_speed_mps", 1.5),
        grace_seconds=rss.get("grace_seconds", 2.0),
    )

    pu = rules_d.get("pushing", {})
    pu_cfg = PushingConfig(
        enabled=pu.get("enabled", True),
        min_closing_speed_mps=pu.get("min_closing_speed_mps", 0.05),
        similar_force_margin_mps=pu.get("similar_force_margin_mps", 0.15),
        persistence_frames=pu.get("persistence_frames", 15),
    )

    cr = rules_d.get("crashing", {})
    cr_cfg = CrashingConfig(
        enabled=cr.get("enabled", True),
        fault_speed_threshold_mps=cr.get("fault_speed_threshold_mps", 1.5),
        both_fault_threshold_mps=cr.get("both_fault_threshold_mps", 0.3),
        retrigger_cooldown_seconds=cr.get("retrigger_cooldown_seconds", 2.0),
    )

    das = rules_d.get("defense_area_stoppage", {})
    das_cfg = DefenseAreaStoppageConfig(
        enabled=das.get("enabled", True),
        min_distance_meters=das.get("min_distance_meters", 0.2),
        grace_seconds=das.get("grace_seconds", 2.0),
    )

    bpi = rules_d.get("ball_placement_interference", {})
    bpi_cfg = BallPlacementInterferenceConfig(
        enabled=bpi.get("enabled", True),
        stadium_radius_meters=bpi.get("stadium_radius_meters", 0.5),
        grace_seconds=bpi.get("grace_seconds", 2.0),
    )

    rules = RulesConfig(
        goal_detection=goal_cfg,
        out_of_bounds=oob_cfg,
        defense_area=da_cfg,
        keep_out=ko_cfg,
        ball_speed=bs_cfg,
        double_touch=dt_cfg,
        keeper_held_ball=khb_cfg,
        excessive_dribbling=ed_cfg,
        robot_stop_speed=rss_cfg,
        pushing=pu_cfg,
        crashing=cr_cfg,
        defense_area_stoppage=das_cfg,
        ball_placement_interference=bpi_cfg,
    )

    game_d = data.get("game", {})
    aa = game_d.get("auto_advance", {})
    stop_to_next_command = aa.get("stop_to_next_command", aa.get("stop_to_prepare_kickoff", True))
    auto_advance = AutoAdvanceConfig(
        stop_to_next_command=stop_to_next_command,
        prepare_kickoff_to_normal=aa.get("prepare_kickoff_to_normal", True),
        prepare_penalty_to_normal=aa.get("prepare_penalty_to_normal", True),
        direct_free_to_normal=aa.get("direct_free_to_normal", True),
        ball_placement_to_next=aa.get("ball_placement_to_next", True),
        normal_start_to_force=aa.get("normal_start_to_force", True),
    )
    game = GameConfig(
        half_duration_seconds=game_d.get("half_duration_seconds", 300.0),
        kickoff_team=game_d.get("kickoff_team", "yellow"),
        force_start_after_goal=game_d.get("force_start_after_goal", False),
        stop_duration_seconds=game_d.get("stop_duration_seconds", 3.0),
        prepare_duration_seconds=game_d.get("prepare_duration_seconds", 3.0),
        kickoff_timeout_seconds=game_d.get("kickoff_timeout_seconds", 10.0),
        auto_advance=auto_advance,
    )

    return RefereeProfile(
        profile_name=data.get("profile_name", "unknown"),
        rules=rules,
        game=game,
    )
