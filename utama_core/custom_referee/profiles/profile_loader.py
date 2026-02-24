"""Profile loader: parses YAML referee profiles into typed dataclasses."""

from __future__ import annotations

import os
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
class GameConfig:
    half_duration_seconds: float = 300.0
    kickoff_team: str = "yellow"
    # If True, skip PREPARE_KICKOFF and issue FORCE_START automatically after
    # stop_duration_seconds.  Used by the arcade profile for continuous play.
    force_start_after_goal: bool = False
    # How long to stay in STOP before auto-advancing (only when
    # force_start_after_goal=True).  Set to 0.0 to advance immediately.
    stop_duration_seconds: float = 3.0


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

    Built-in names: "strict_ai", "exhibition", "arcade".
    """
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
    game = GameConfig(
        half_duration_seconds=game_d.get("half_duration_seconds", 300.0),
        kickoff_team=game_d.get("kickoff_team", "yellow"),
        force_start_after_goal=game_d.get("force_start_after_goal", False),
        stop_duration_seconds=game_d.get("stop_duration_seconds", 3.0),
    )

    return RefereeProfile(
        profile_name=data.get("profile_name", "unknown"),
        geometry=geometry,
        rules=rules,
        game=game,
    )
