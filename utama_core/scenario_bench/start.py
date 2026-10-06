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

import dataclasses
import json
import math
import random
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
    OPEN_PLAY_COUNTER = "open_play_counter"  # just lost the ball; also harvested FORCE_START restarts
    OPEN_PLAY_POSSESSION = "open_play_possession"  # our possession, a pass still to make
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

    def to_dict(self) -> dict:
        """Plain-JSON representation for a persisted bank (see `save_bank`).

        Field state only — no tactic `mem`, matching the module docstring's
        policy-agnostic constraint. `source_replay` is stored as a string
        (or None) since it's provenance metadata, not a path this process
        needs to resolve back into a live `Path` for anything other than
        display.
        """
        s = self.scenario
        return {
            "scenario_id": self.scenario_id,
            "lifecycle": self.lifecycle.value,
            "lead_in_s": self.lead_in_s,
            "provenance": {
                "source_run_id": self.provenance.source_run_id,
                "evaluator_version": self.provenance.evaluator_version,
                "trigger": self.provenance.trigger.value,
                "family": self.provenance.family.value,
                "anchor_tick": self.provenance.anchor_tick,
                "source_replay": str(self.provenance.source_replay) if self.provenance.source_replay else None,
                "perspective": self.provenance.perspective,
            },
            "scenario": {
                "sim_time": s.sim_time,
                "ball_x": s.ball_x,
                "ball_y": s.ball_y,
                "ball_vx": s.ball_vx,
                "ball_vy": s.ball_vy,
                "friendly_robots": [dataclasses.asdict(r) for r in s.friendly_robots],
                "enemy_robots": [dataclasses.asdict(r) for r in s.enemy_robots],
                "referee_command": s.referee_command.name if s.referee_command is not None else None,
                "config_a_name": s.config_a_name,
                "config_b_name": s.config_b_name,
                "frame_ts": s.frame_ts,
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "BenchScenario":
        """Inverse of `to_dict`. `source_replay` round-trips to a `Path`
        wrapping whatever string was stored (or `Path(".")` when it was
        None, matching `Scenario.source_replay`'s non-Optional type — the
        original replay a persisted scenario came from is provenance, not a
        file this process needs to still exist)."""
        sd = d["scenario"]
        prov = d["provenance"]
        scenario = Scenario(
            sim_time=sd["sim_time"],
            ball_x=sd["ball_x"],
            ball_y=sd["ball_y"],
            ball_vx=sd["ball_vx"],
            ball_vy=sd["ball_vy"],
            friendly_robots=tuple(RobotState(**r) for r in sd["friendly_robots"]),
            enemy_robots=tuple(RobotState(**r) for r in sd["enemy_robots"]),
            referee_command=RefereeCommand[sd["referee_command"]] if sd["referee_command"] is not None else None,
            source_replay=Path(prov["source_replay"]) if prov["source_replay"] else Path("."),
            config_a_name=sd.get("config_a_name"),
            config_b_name=sd.get("config_b_name"),
            frame_ts=sd.get("frame_ts", 0.0),
        )
        provenance = ScenarioProvenance(
            source_run_id=prov["source_run_id"],
            evaluator_version=prov["evaluator_version"],
            trigger=ScenarioTrigger(prov["trigger"]),
            family=ScenarioFamily(prov["family"]),
            anchor_tick=prov["anchor_tick"],
            source_replay=Path(prov["source_replay"]) if prov["source_replay"] else None,
            perspective=prov["perspective"],
        )
        return BenchScenario(
            scenario_id=d["scenario_id"],
            scenario=scenario,
            provenance=provenance,
            lifecycle=ScenarioLifecycle(d["lifecycle"]),
            lead_in_s=d.get("lead_in_s", 0.0),
        )


@dataclass(frozen=True)
class StaticScreenResult:
    ok: bool
    violations: tuple[str, ...] = field(default_factory=tuple)


def _in_bounds(x: float, y: float, margin: float = _OUT_OF_BOUNDS_MARGIN_M) -> bool:
    half_len = STANDARD_FIELD_DIMS.full_field_half_length + margin
    half_width = STANDARD_FIELD_DIMS.full_field_half_width + margin
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
        # no margin: the sim refuses to teleport a robot past the field lines, and a
        # start that can't be set up would be scored as if nothing happened
        if not _in_bounds(robot.x, robot.y, margin=0.0):
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


# rsim is deterministic, so one run per scenario is one sample and says nothing
# about noise. `jittered` gives each repeat of a scenario a slightly different
# start: robots moved up to this far, and turned up to this much.
_JITTER_POS_M = 0.05
_JITTER_OREN_RAD = 0.1
# Robots this close to the ball keep their exact pose: their relation to the ball
# (on the dribbler, lined up to take a restart) is what the scenario is about.
_JITTER_KEEP_NEAR_BALL_M = 0.3


def jittered(bench_scenario: "BenchScenario", seed: int) -> "BenchScenario":
    """`bench_scenario` with every robot not near the ball nudged by a small random
    offset, reproducible from (`scenario_id`, `seed`). Seed 0 is the scenario as
    authored. A draw that fails `static_screen` (an overlap) is redrawn."""
    if seed == 0:
        return bench_scenario
    rng = random.Random(f"{bench_scenario.scenario_id}:{seed}")
    sc = bench_scenario.scenario

    def nudge(r: RobotState) -> RobotState:
        if math.hypot(r.x - sc.ball_x, r.y - sc.ball_y) < _JITTER_KEEP_NEAR_BALL_M:
            return r
        return dataclasses.replace(
            r,
            x=r.x + rng.uniform(-_JITTER_POS_M, _JITTER_POS_M),
            y=r.y + rng.uniform(-_JITTER_POS_M, _JITTER_POS_M),
            orientation=r.orientation + rng.uniform(-_JITTER_OREN_RAD, _JITTER_OREN_RAD),
        )

    for _ in range(20):
        candidate = dataclasses.replace(
            sc,
            friendly_robots=tuple(nudge(r) for r in sc.friendly_robots),
            enemy_robots=tuple(nudge(r) for r in sc.enemy_robots),
        )
        if static_screen(candidate).ok:
            return dataclasses.replace(bench_scenario, scenario=candidate)
    return bench_scenario


# Bumped whenever `BenchScenario.to_dict`'s schema changes in a way old
# banks can't be read back with (a field renamed/removed, not just added) —
# `load_bank` refuses to load a mismatched major version rather than
# silently misinterpreting an old file.
_BANK_SCHEMA_VERSION = 1


def save_bank(scenarios: list["BenchScenario"], path: Path, *, bank_id: str) -> None:
    """Write `scenarios` to `path` as a single JSON manifest.

    Per roadmap item 14: "Immutable per bank version — adding scenarios
    makes a new bank ID and forces a champion re-baseline." `bank_id` is
    caller-chosen (e.g. "v1", or a date-stamped tag) and stored alongside a
    schema version and scenario count so a stale/mismatched bank fails
    loudly on load rather than being silently treated as compatible. This
    function does not enforce immutability itself (nothing stops a second
    `save_bank` call to the same path) — that discipline is a caller/process
    convention (new scenarios -> new path/bank_id), not a file-format one.
    """
    payload = {
        "bank_id": bank_id,
        "schema_version": _BANK_SCHEMA_VERSION,
        "n_scenarios": len(scenarios),
        "scenarios": [s.to_dict() for s in scenarios],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_bank(path: Path) -> tuple[str, list["BenchScenario"]]:
    """Inverse of `save_bank`. Returns `(bank_id, scenarios)`.

    Raises `ValueError` if the file's `schema_version` doesn't match this
    process's `_BANK_SCHEMA_VERSION` — a version bump means the dict shape
    changed incompatibly, and guessing at a mismatched shape would produce
    a wrong-but-not-crashing bank silently, which is exactly the failure
    mode item 14's contamination-audit design exists to avoid.
    """
    payload = json.loads(path.read_text())
    version = payload.get("schema_version")
    if version != _BANK_SCHEMA_VERSION:
        raise ValueError(
            f"Bank {path} has schema_version={version}, this process expects "
            f"{_BANK_SCHEMA_VERSION}. Re-harvest and re-save rather than loading it as-is."
        )
    scenarios = [BenchScenario.from_dict(d) for d in payload["scenarios"]]
    return payload["bank_id"], scenarios


# Two starts this close are the same situation twice (kickoff formations recur match after
# match): keeping both adds count to a bank, not independent evidence, and inflates its t.
_DUPLICATE_BALL_M = 0.10
_DUPLICATE_ROBOT_M = 0.15


def _positions(scenario: Scenario) -> dict[tuple[str, int], tuple[float, float]]:
    return {("friendly", r.id): (r.x, r.y) for r in scenario.friendly_robots} | {
        ("enemy", r.id): (r.x, r.y) for r in scenario.enemy_robots
    }


def is_near_duplicate(a: "BenchScenario", b: "BenchScenario") -> bool:
    """Same family, perspective and referee command, ball within `_DUPLICATE_BALL_M` and
    every robot (by team and id) within `_DUPLICATE_ROBOT_M`."""
    sa, sb = a.scenario, b.scenario
    if (a.provenance.family, a.provenance.perspective, sa.referee_command) != (
        b.provenance.family,
        b.provenance.perspective,
        sb.referee_command,
    ):
        return False
    if math.dist((sa.ball_x, sa.ball_y), (sb.ball_x, sb.ball_y)) > _DUPLICATE_BALL_M:
        return False
    pa, pb = _positions(sa), _positions(sb)
    return pa.keys() == pb.keys() and all(math.dist(pa[k], pb[k]) <= _DUPLICATE_ROBOT_M for k in pa)


def drop_near_duplicates(
    scenarios: list["BenchScenario"], *, keep: list["BenchScenario"] = ()
) -> list["BenchScenario"]:
    """`scenarios` minus near-duplicates of `keep` (an existing bank) or of an earlier
    scenario in the list. A survivor whose id is already taken gets its source run appended."""
    kept = list(keep)
    ids = {s.scenario_id for s in kept}
    new: list[BenchScenario] = []
    for s in scenarios:
        if any(is_near_duplicate(s, k) for k in kept):
            continue
        if s.scenario_id in ids:
            s = dataclasses.replace(s, scenario_id=f"{s.scenario_id}_{s.provenance.source_run_id}")
        ids.add(s.scenario_id)
        kept.append(s)
        new.append(s)
    return new


# --- Situation tags (docs/pitch_zones.md) ------------------------------------------------
#
# Where a start is and who it belongs to, from the candidate's point of view, so a start can
# be found by situation (`scenario_bench.py --where`). Derived from the start's own state, not
# stored in the bank. The candidate is friendly: yellow, defending the right goal, attacking -x.

_ON_BALL_M = 0.2  # a robot centre this close to the ball has it
NEAR_BALL_M = 1.5  # robots this close to the ball count in the `near` tag
TAGS = ("third", "lane", "restart", "ball", "near")


def start_tags(bench_scenario: "BenchScenario") -> dict[str, str]:
    """`third`: defensive / middle / attacking. `lane`: left_wing / centre / right_wing (as the
    candidate faces the goal it attacks). `restart`: ours / theirs (the team the referee command
    gives the kickoff, free kick, penalty or ball placement to), stop, or live. `ball`: ours /
    theirs (a robot within `_ON_BALL_M`) or loose. `near`: our and their robots within
    `NEAR_BALL_M` of the ball, e.g. "2v3"."""
    s = bench_scenario.scenario
    half_length = STANDARD_FIELD_DIMS.full_field_half_length
    progress = half_length - s.ball_x  # metres from our goal line toward the goal we attack
    third = 2 * half_length / 3
    left = -s.ball_y  # facing -x, our left is -y
    centre = STANDARD_FIELD_DIMS.half_defense_area_width

    cmd = s.referee_command
    if cmd is None or cmd in (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START):
        restart = "live"
    elif cmd.name.endswith("_YELLOW"):
        restart = "ours"
    elif cmd.name.endswith("_BLUE"):
        restart = "theirs"
    else:
        restart = "stop"

    def dists(robots):
        return [math.hypot(r.x - s.ball_x, r.y - s.ball_y) for r in robots]

    ours, theirs = dists(s.friendly_robots), dists(s.enemy_robots)
    nearest_ours, nearest_theirs = min(ours, default=math.inf), min(theirs, default=math.inf)
    if min(nearest_ours, nearest_theirs) > _ON_BALL_M:
        ball = "loose"
    else:
        ball = "ours" if nearest_ours <= nearest_theirs else "theirs"

    return {
        "third": "defensive" if progress < third else "middle" if progress < 2 * third else "attacking",
        "lane": "left_wing" if left > centre else "right_wing" if left < -centre else "centre",
        "restart": restart,
        "ball": ball,
        "near": f"{sum(d <= NEAR_BALL_M for d in ours)}v{sum(d <= NEAR_BALL_M for d in theirs)}",
    }


def matches_where(tags: dict[str, str], where: dict[str, set[str]]) -> bool:
    """`where` maps a tag to the values it may take (`--where third=defensive ball=theirs,loose`)."""
    return all(tags[key] in values for key, values in where.items())
