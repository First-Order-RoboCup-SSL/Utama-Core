"""Scenario bench data model: provenance, lifecycle, and the static validity
screen shared by every scenario source (hand-authored, harvested, weakness,
event-triggered — see `docs/roadmap.md` item 14's outer-loop fast half).

This module intentionally does not know how to *run* a scenario — that's
`scenario.apply_scenario` (which `BenchScenario.to_scenario()` feeds into)
plus whatever match-loop the bench tool itself drives. This module only
answers: is this snapshot well-formed, and where did it come from.

Design constraints carried over from the design conversation that preceded
this module (2026-09-04), not just this file's own choices:
- The bank must be policy-agnostic. `BenchScenario` stores field state only
  (positions/velocities/referee command), never tactic `mem` — the same
  limitation `scenario.py` already documents for repro use. A candidate
  strategy that restructures its tactic layer must still be able to consume
  every scenario in the bank.
- Every scenario carries enough provenance to retire it *by provenance* in
  one query if a harvest run is later found to be contaminated (see item 15
  and this session's discovery that a 925-file "complete" replay run was
  entirely pre-fix data stuck at kickoff).
- Lifecycle is deliberately conservative: `CANDIDATE` scenarios never score.
  Only `ACTIVE` scenarios do. Nothing enters the scored bank silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import RobotState, Scenario

# A robot/ball placed further than this outside the nominal field rectangle
# is not "just off by a little", it's a fabrication or coordinate-frame bug.
_OUT_OF_BOUNDS_MARGIN_M = 0.5

# Two robots (or a robot and the ball) closer than this are overlapping,
# not just close — 2x ROBOT_RADIUS plus a small slack for float noise.
_MIN_ROBOT_SEPARATION_M = 2 * ROBOT_RADIUS - 0.01

# Hand-authored scenarios don't have a previous tick to compare against, so
# the physical-plausibility speed check only applies when a scenario states
# nonzero velocity. Generous — this is a sanity check, not a controller limit.
_MAX_PLAUSIBLE_SPEED_MPS = 6.0


class ScenarioTrigger(Enum):
    """How this scenario's anchor moment was identified."""

    HAND_AUTHORED = "hand_authored"
    RESTART = "restart"  # referee transition into live play (see item 14)
    EVENT = "event"  # MatchStats-detected: possession change, entry, etc.
    PERIODIC = "periodic"  # fallback sampling, not trigger-detected


class ScenarioFamily(Enum):
    """Coarse situational grouping, used for frequency weighting and
    per-family score aggregation (never a global composite — see roadmap
    item 14's "per-family is the unit" scoring note)."""

    KICKOFF = "kickoff"
    DIRECT_FREE_ATTACKING = "direct_free_attacking"
    DIRECT_FREE_DEFENDING = "direct_free_defending"
    PENALTY = "penalty"
    BALL_PLACEMENT = "ball_placement"
    OPEN_PLAY_COUNTER = "open_play_counter"
    OPEN_PLAY_LOOSE_BALL = "open_play_loose_ball"
    WEAKNESS = "weakness"  # harvested from a lost play; tagged separately


class ScenarioLifecycle(Enum):
    """Only ACTIVE scenarios are scored. See module docstring."""

    CANDIDATE = "candidate"  # freshly harvested/authored, not yet screened
    VALIDATED = "validated"  # passed static + dynamic screen + spot-check
    ACTIVE = "active"  # in the scored bank
    RETIRED = "retired"  # excluded; provenance kept for audit


@dataclass(frozen=True)
class ScenarioProvenance:
    """Where a scenario came from, sufficient to retire an entire batch by
    query if its source run is later found to be contaminated."""

    source_run_id: str  # e.g. a tournament directory name, or "hand_authored"
    evaluator_version: str  # git revision the source run (or authoring) used
    trigger: ScenarioTrigger
    family: ScenarioFamily
    anchor_tick: Optional[float] = None  # sim_time of the anchor moment; None for hand-authored
    source_replay: Optional[Path] = None
    perspective: str = "candidate_kicking"  # or "candidate_defending" — restarts are asymmetric (see item 14)


@dataclass(frozen=True)
class BenchScenario:
    """A single scenario in the bench, wrapping the field-state `Scenario`
    the runner actually consumes with provenance + lifecycle metadata.

    `scenario_id` is a short stable slug (e.g. "kickoff_center_v1") used for
    per-scenario quality tracking across bank versions (roadmap item 14's
    "ongoing quality tracking" — whether a scenario's bench outcome tracked
    the full-game outcome after each ladder run).
    """

    scenario_id: str
    scenario: Scenario
    provenance: ScenarioProvenance
    lifecycle: ScenarioLifecycle = ScenarioLifecycle.CANDIDATE
    lead_in_s: float = 0.0  # event-triggered scenarios start lead_in_s before the anchor (see item 14 sec 3)

    def to_scenario(self) -> Scenario:
        return self.scenario


@dataclass(frozen=True)
class StaticScreenResult:
    ok: bool
    violations: tuple[str, ...] = field(default_factory=tuple)


def _in_bounds(x: float, y: float) -> bool:
    half_len = STANDARD_FIELD_DIMS.full_field_half_length + _OUT_OF_BOUNDS_MARGIN_M
    half_width = STANDARD_FIELD_DIMS.full_field_half_width + _OUT_OF_BOUNDS_MARGIN_M
    return -half_len <= x <= half_len and -half_width <= y <= half_width


def _speed(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)


def static_screen(scenario: Scenario) -> StaticScreenResult:
    """Cheap, self-contained checks on a single snapshot — no play-forward.

    Mirrors roadmap item 14's static-check list: in-bounds, no overlap,
    physically plausible speeds, at least one robot per team present. Does
    NOT check "ball not stationary too long" or "no stall in surrounding
    window" — those need the source replay's surrounding ticks, which this
    function deliberately doesn't take, so those checks belong to whatever
    harvester builds the `BenchScenario` (it already has the replay open).
    """
    violations: list[str] = []

    if not _in_bounds(scenario.ball_x, scenario.ball_y):
        violations.append(f"ball out of bounds at ({scenario.ball_x:.2f}, {scenario.ball_y:.2f})")
    if _speed(scenario.ball_vx, scenario.ball_vy) > _MAX_PLAUSIBLE_SPEED_MPS * 2:
        # Balls legitimately travel faster than robots (a hard kick), so this
        # gets a looser multiplier than the robot check below.
        violations.append(f"implausible ball speed {_speed(scenario.ball_vx, scenario.ball_vy):.2f} m/s")

    all_robots: list[tuple[str, RobotState]] = [("friendly", r) for r in scenario.friendly_robots] + [
        ("enemy", r) for r in scenario.enemy_robots
    ]

    if not scenario.friendly_robots:
        violations.append("no friendly robots present")
    if not scenario.enemy_robots:
        violations.append("no enemy robots present")

    for side, robot in all_robots:
        if not _in_bounds(robot.x, robot.y):
            violations.append(f"{side} robot {robot.id} out of bounds at ({robot.x:.2f}, {robot.y:.2f})")
        speed = _speed(robot.vx, robot.vy)
        if speed > _MAX_PLAUSIBLE_SPEED_MPS:
            violations.append(f"{side} robot {robot.id} implausible speed {speed:.2f} m/s")

    for i in range(len(all_robots)):
        side_i, robot_i = all_robots[i]
        for j in range(i + 1, len(all_robots)):
            side_j, robot_j = all_robots[j]
            dist = math.hypot(robot_i.x - robot_j.x, robot_i.y - robot_j.y)
            if dist < _MIN_ROBOT_SEPARATION_M:
                violations.append(
                    f"{side_i} robot {robot_i.id} overlaps {side_j} robot {robot_j.id} "
                    f"(dist={dist:.3f}m < {_MIN_ROBOT_SEPARATION_M:.3f}m)"
                )

    return StaticScreenResult(ok=not violations, violations=tuple(violations))
