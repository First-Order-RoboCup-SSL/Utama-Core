"""Profile loader: parses YAML referee profiles into typed dataclasses."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from utama_core.custom_referee.geometry import RefereeGeometry

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
class RulesConfig:
    goal_detection: GoalDetectionConfig = field(default_factory=GoalDetectionConfig)
    out_of_bounds: OutOfBoundsConfig = field(default_factory=OutOfBoundsConfig)
    defense_area: DefenseAreaConfig = field(default_factory=DefenseAreaConfig)
    keep_out: KeepOutConfig = field(default_factory=KeepOutConfig)


# ---------------------------------------------------------------------------
# Game config
# ---------------------------------------------------------------------------


@dataclass
class AutoAdvanceConfig:
    """Controls which state-machine transitions fire automatically.

    Set all to False for physical environments where a human operator must
    explicitly advance the state to prevent robots from moving unexpectedly.
    """

    # STOP → PREPARE_KICKOFF_* when all robots have cleared the ball.
    stop_to_prepare_kickoff: bool = True
    # PREPARE_KICKOFF_* → NORMAL_START after prepare_duration_seconds when
    # the kicker is inside the centre circle.
    prepare_kickoff_to_normal: bool = True
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
    geometry: RefereeGeometry
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
    geo_d = data.get("geometry", {})
    geometry = RefereeGeometry(
        half_length=geo_d.get("half_length", 4.5),
        half_width=geo_d.get("half_width", 3.0),
        half_goal_width=geo_d.get("half_goal_width", 0.5),
        half_defense_length=geo_d.get("half_defense_length", 0.5),
        half_defense_width=geo_d.get("half_defense_width", 1.0),
        center_circle_radius=geo_d.get("center_circle_radius", 0.5),
    )

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

    rules = RulesConfig(
        goal_detection=goal_cfg,
        out_of_bounds=oob_cfg,
        defense_area=da_cfg,
        keep_out=ko_cfg,
    )

    game_d = data.get("game", {})
    aa = game_d.get("auto_advance", {})
    auto_advance = AutoAdvanceConfig(
        stop_to_prepare_kickoff=aa.get("stop_to_prepare_kickoff", True),
        prepare_kickoff_to_normal=aa.get("prepare_kickoff_to_normal", True),
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
        geometry=geometry,
        rules=rules,
        game=game,
    )
