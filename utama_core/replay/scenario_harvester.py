"""Harvest restart-triggered `BenchScenario`s from a tagged tournament run.

Implements the "restart is the moment agency starts" design from
`docs/roadmap.md` item 14 (2026-09-04 design pass): a scenario's anchor is
the transition INTO live play, not the PREPARE/STOP state that precedes it —
`PREPARE_KICKOFF_*/PREPARE_PENALTY_* → NORMAL_START`,
`STOP → DIRECT_FREE_*`, `STOP → FORCE_START`. Every restart yields two
`BenchScenario`s (candidate kicking, candidate defending) — restarts are
asymmetric.

Match-level trust gate, per that same design pass: a match's replay is only
harvested from when its `<match_tag>.stats.json` (written by
`tournament_lib.run_match`'s `stats_path`, see `MatchStats.to_json`) exists
and reports `stall_events == []`. A match with no stats file, or any stall
event, contributes nothing — this is the gate that would have caught the
925-file pre-fix contaminated replay run this session found (every one of
those matches was stuck at `PREPARE_KICKOFF_YELLOW` from t=0, so it either
has a `RESTART_STALL` recorded or, for even older runs with no watchdog at
all, no `.stats.json` to check — both fail closed here).

Open play (`open_play_per_match`, trigger `EVENT`): `turnover_breakdown.analyse_match`
already follows every pass and real ball loss of the yellow side (the candidate), so each
becomes a start: `OPEN_PLAY_POSSESSION` `_PASS_LEAD_S` before a pass is released (the pass is
still the candidate's to make), `OPEN_PLAY_COUNTER` at a real loss (the candidate must defend
the counter). Only moments inside unbroken live play count, and at most N of each per match
are kept, since a match has dozens and they are costly to screen.

This module does NOT run a tournament itself — `harvest_run_dir` takes an
already-completed run directory (e.g. `replays/tournament_.../`) and is a
pure read; see `tools/scenario_bench.py` for the harvest → screen → score →
report pipeline this feeds.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.bench_scenario import (
    BenchScenario,
    ScenarioFamily,
    ScenarioProvenance,
    ScenarioTrigger,
    static_screen,
)
from utama_core.replay.scenario import Scenario, scenario_from_replay
from utama_core.replay.turnover_breakdown import _is_real, analyse_match

logger = logging.getLogger(__name__)

# Restart families keyed by the referee command that is transitioned INTO.
# `perspective` is candidate-relative and resolved per-match against which
# colour the candidate is playing — see `_family_and_perspective`.
_KICKOFF_COMMANDS = frozenset({RefereeCommand.NORMAL_START})
_PENALTY_PREV_COMMANDS = frozenset({RefereeCommand.PREPARE_PENALTY_YELLOW, RefereeCommand.PREPARE_PENALTY_BLUE})
_KICKOFF_PREV_COMMANDS = frozenset({RefereeCommand.PREPARE_KICKOFF_YELLOW, RefereeCommand.PREPARE_KICKOFF_BLUE})
_DIRECT_FREE_COMMANDS = frozenset({RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.DIRECT_FREE_BLUE})
_LIVE_PLAY_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})
# A free kick follows STOP, or ball placement when the ball had to be moved first
# (`CustomReferee`'s usual sequence after the ball goes out).
_DIRECT_FREE_PREV_COMMANDS = frozenset(
    {RefereeCommand.STOP, RefereeCommand.BALL_PLACEMENT_YELLOW, RefereeCommand.BALL_PLACEMENT_BLUE}
)

# A restart ceremony resolving in under this many sim seconds after the
# previous transition is treated as noise (e.g. two sidecar rows for the
# same instant) rather than two independent restarts.
_MIN_RESTART_GAP_S = 0.5


@dataclass(frozen=True)
class RestartTransition:
    """One detected transition into live play, from a `.intentions.jsonl`
    sidecar's referee-event rows."""

    sim_time: float
    prev_command: Optional[RefereeCommand]
    command: RefereeCommand
    # Which colour (True=yellow) the transition's restart favours — the
    # kicking/placed team, derived from `prev_command`/`command` where
    # possible (DIRECT_FREE_YELLOW/BLUE encode it directly; a plain
    # NORMAL_START/FORCE_START from STOP does not, so this is None there
    # and perspective falls back to "whoever the ball is nearest" — the
    # harvester doesn't have ball position at this layer, so that
    # fallback is resolved later via the loaded `Scenario`).
    kicking_is_yellow: Optional[bool]


def _referee_rows(sidecar_path: Path) -> list[dict]:
    if not sidecar_path.exists():
        return []
    rows = []
    with open(sidecar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "referee":
                continue
            rows.append(row)
    return rows


def _kicking_is_yellow(command: RefereeCommand) -> Optional[bool]:
    if command in (RefereeCommand.DIRECT_FREE_YELLOW,):
        return True
    if command in (RefereeCommand.DIRECT_FREE_BLUE,):
        return False
    return None


def find_restart_transitions(sidecar_path: Path) -> list[RestartTransition]:
    """Scan a `.intentions.jsonl` sidecar for transitions into live play or
    into a DIRECT_FREE command, in the order the design calls out:
    `PREPARE_KICKOFF_*/PREPARE_PENALTY_* -> NORMAL_START`,
    `STOP/BALL_PLACEMENT_* -> DIRECT_FREE_*`, `STOP -> FORCE_START`.

    Returns transitions in ascending `sim_time` order. A transition whose
    `sim_time` is within `_MIN_RESTART_GAP_S` of the previous one found is
    dropped as sidecar noise, not a second independent restart.
    """
    rows = _referee_rows(sidecar_path)
    transitions: list[RestartTransition] = []
    prev_command: Optional[RefereeCommand] = None
    last_transition_ts = float("-inf")

    for row in rows:
        command_name = row.get("command")
        ts = row.get("sim_time")
        if command_name is None or ts is None:
            continue
        try:
            command = RefereeCommand[command_name]
        except KeyError:
            continue

        is_restart_into_live = (
            command in _KICKOFF_COMMANDS and prev_command in (_KICKOFF_PREV_COMMANDS | _PENALTY_PREV_COMMANDS)
        ) or (command == RefereeCommand.FORCE_START and prev_command == RefereeCommand.STOP)
        is_restart_into_direct_free = command in _DIRECT_FREE_COMMANDS and prev_command in _DIRECT_FREE_PREV_COMMANDS

        if (is_restart_into_live or is_restart_into_direct_free) and (ts - last_transition_ts) >= _MIN_RESTART_GAP_S:
            transitions.append(
                RestartTransition(
                    sim_time=float(ts),
                    prev_command=prev_command,
                    command=command,
                    kicking_is_yellow=_kicking_is_yellow(command),
                )
            )
            last_transition_ts = ts

        prev_command = command

    return transitions


def _family_for(transition: RestartTransition) -> ScenarioFamily:
    if transition.command in _KICKOFF_COMMANDS and transition.prev_command in _KICKOFF_PREV_COMMANDS:
        return ScenarioFamily.KICKOFF
    if transition.command in _KICKOFF_COMMANDS and transition.prev_command in _PENALTY_PREV_COMMANDS:
        return ScenarioFamily.PENALTY
    if transition.command in _DIRECT_FREE_COMMANDS:
        # Perspective (attacking/defending) is resolved per-candidate by the
        # caller (`_perspective_family`), not here — a DIRECT_FREE transition
        # alone doesn't know which side is "the candidate".
        return ScenarioFamily.DIRECT_FREE_ATTACKING
    return ScenarioFamily.OPEN_PLAY_COUNTER  # FORCE_START from STOP with no free-kick context


def _perspective_family(
    transition: RestartTransition, family: ScenarioFamily, candidate_is_yellow: bool
) -> tuple[ScenarioFamily, str]:
    if family != ScenarioFamily.DIRECT_FREE_ATTACKING:
        return family, "candidate_kicking"
    if transition.kicking_is_yellow is None:
        return family, "candidate_kicking"
    candidate_is_kicking = transition.kicking_is_yellow == candidate_is_yellow
    if candidate_is_kicking:
        return ScenarioFamily.DIRECT_FREE_ATTACKING, "candidate_kicking"
    return ScenarioFamily.DIRECT_FREE_DEFENDING, "candidate_defending"


def match_is_trustworthy(stats_path: Path) -> bool:
    """The match-level harvest gate: a `.stats.json` must exist and report
    zero stall events. Missing or unparseable files fail closed (not
    trustworthy) — see module docstring."""
    if not stats_path.exists():
        return False
    try:
        data = json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    stall_events = data.get("stall_events")
    return stall_events is not None and len(stall_events) == 0


def _stats_path_for(replay_path: Path) -> Path:
    return replay_path.with_name(f"{replay_path.stem}.stats.json")


def _sidecar_path_for(replay_path: Path) -> Path:
    return replay_path.with_name(f"{replay_path.stem}.intentions.jsonl")


_PASS_LEAD_S = 1.0


@dataclass(frozen=True)
class OpenPlayEvent:
    sim_time: float  # where the scenario starts
    family: ScenarioFamily
    perspective: str


def _live_throughout(referee_rows: list[dict], t0: float, t1: float) -> bool:
    """Live play from `t0` to `t1`: the last command at or before `t0` is live and none follows by `t1`."""
    command = None
    for row in referee_rows:
        ts = row.get("sim_time")
        if ts is None:
            continue
        if ts <= t0:
            command = row.get("command")
        elif ts <= t1:
            return False
    return command is not None and RefereeCommand[command] in _LIVE_PLAY_COMMANDS


def open_play_events(analysis: dict, referee_rows: list[dict]) -> list[OpenPlayEvent]:
    """Open-play starts from one `turnover_breakdown.analyse_match` result (see module docstring)."""
    events = [
        OpenPlayEvent(p["t"] - _PASS_LEAD_S, ScenarioFamily.OPEN_PLAY_POSSESSION, "candidate_in_possession")
        for p in analysis["passes"]
        if _live_throughout(referee_rows, p["t"] - _PASS_LEAD_S, p["t"])
    ]
    events += [
        OpenPlayEvent(t["t"], ScenarioFamily.OPEN_PLAY_COUNTER, "candidate_defending")
        for t in analysis["turnovers"]
        if _is_real(t) and _live_throughout(referee_rows, t["t"], t["t"])
    ]
    return events


def pick_events(events: list[OpenPlayEvent], per_family: int, *, seed: str) -> list[OpenPlayEvent]:
    """At most `per_family` of each family, drawn reproducibly by `seed` (the match name), in time order."""
    rng = random.Random(seed)
    picked = []
    for family in dict.fromkeys(e.family for e in events):
        same = [e for e in events if e.family == family]
        picked += rng.sample(same, min(per_family, len(same)))
    return sorted(picked, key=lambda e: e.sim_time)


def _harvested(replay_path: Path, t: float) -> Optional[Scenario]:
    """The replay's field state at `t`, or None if it can't be read or fails the static screen."""
    try:
        scenario = scenario_from_replay(replay_path, t)
    except ValueError:
        logger.warning("Could not load frame at t=%.3f from %s", t, replay_path)
        return None
    screen = static_screen(scenario)
    if not screen.ok:
        logger.info(
            "Dropping start at t=%.3f in %s: static screen failed (%s)", t, replay_path, "; ".join(screen.violations)
        )
        return None
    return scenario


def harvest_replay(
    replay_path: Path,
    *,
    source_run_id: str,
    evaluator_version: str,
    candidate_is_yellow: bool = True,
    open_play_per_match: int = 0,
) -> list[BenchScenario]:
    """Harvest every restart-transition `BenchScenario` from one replay, plus up to
    `open_play_per_match` open-play starts of each kind (see module docstring).

    Does NOT check `match_is_trustworthy` itself — callers (typically
    `harvest_run_dir`) are expected to have already gated on it, since
    checking per-replay here would silently produce an empty list for an
    untrustworthy match instead of letting the caller report *why* nothing
    was harvested.
    """
    sidecar_path = _sidecar_path_for(replay_path)
    transitions = find_restart_transitions(sidecar_path)
    scenarios: list[BenchScenario] = []

    for index, transition in enumerate(transitions):
        # The opening kickoff is the same situation in every match (both teams in their
        # kickoff formation): harvested from a whole run it swamps the bank.
        if index == 0 and _family_for(transition) == ScenarioFamily.KICKOFF:
            continue
        scenario = _harvested(replay_path, transition.sim_time)
        if scenario is None:
            continue

        base_family = _family_for(transition)
        family, perspective = _perspective_family(transition, base_family, candidate_is_yellow)

        scenario_id = f"{replay_path.stem}_t{transition.sim_time:.1f}_{perspective}"
        provenance = ScenarioProvenance(
            source_run_id=source_run_id,
            evaluator_version=evaluator_version,
            trigger=ScenarioTrigger.RESTART,
            family=family,
            anchor_tick=transition.sim_time,
            source_replay=replay_path,
            perspective=perspective,
        )
        scenarios.append(BenchScenario(scenario_id=scenario_id, scenario=scenario, provenance=provenance))

    # analyse_match follows the yellow side only
    if open_play_per_match > 0 and candidate_is_yellow and replay_path.suffix == ".npz":
        events = open_play_events(analyse_match(str(replay_path)), _referee_rows(sidecar_path))
        for event in pick_events(events, open_play_per_match, seed=replay_path.stem):
            scenario = _harvested(replay_path, event.sim_time)
            if scenario is None:
                continue
            provenance = ScenarioProvenance(
                source_run_id=source_run_id,
                evaluator_version=evaluator_version,
                trigger=ScenarioTrigger.EVENT,
                family=event.family,
                anchor_tick=event.sim_time,
                source_replay=replay_path,
                perspective=event.perspective,
            )
            scenario_id = f"{replay_path.stem}_t{event.sim_time:.1f}_{event.family.value}"
            scenarios.append(BenchScenario(scenario_id=scenario_id, scenario=scenario, provenance=provenance))

    return scenarios


def _replay_files_in(run_dir: Path) -> list[Path]:
    # `<match>.sparse_referee.pkl` sits next to each replay; it is not a replay itself.
    pkls = [p for p in run_dir.glob("*.pkl") if not p.name.endswith(".sparse_referee.pkl")]
    return sorted({*run_dir.glob("*.npz"), *pkls})


def harvest_run_dir(
    run_dir: Path,
    *,
    evaluator_version: str,
    candidate_is_yellow: bool = True,
    open_play_per_match: int = 0,
) -> tuple[list[BenchScenario], dict[str, int]]:
    """Harvest restart-triggered scenarios from every trustworthy match in
    `run_dir` (a completed `smoke_tournament.py` run directory).

    Returns `(scenarios, report)` where `report` is a small summary dict
    (`matches_seen`, `matches_trusted`, `matches_untrusted`,
    `scenarios_harvested`) — a caller (the bench CLI) uses this to print
    "harvested N scenarios from M/K trustworthy matches" rather than only
    a bare list.
    """
    replay_paths = _replay_files_in(run_dir)
    source_run_id = run_dir.name

    scenarios: list[BenchScenario] = []
    matches_trusted = 0
    matches_untrusted = 0

    for replay_path in replay_paths:
        stats_path = _stats_path_for(replay_path)
        if not match_is_trustworthy(stats_path):
            matches_untrusted += 1
            logger.info("Skipping untrusted match %s (missing/nonzero-stall .stats.json)", replay_path.name)
            continue
        matches_trusted += 1
        scenarios.extend(
            harvest_replay(
                replay_path,
                source_run_id=source_run_id,
                evaluator_version=evaluator_version,
                candidate_is_yellow=candidate_is_yellow,
                open_play_per_match=open_play_per_match,
            )
        )

    report = {
        "matches_seen": len(replay_paths),
        "matches_trusted": matches_trusted,
        "matches_untrusted": matches_untrusted,
        "scenarios_harvested": len(scenarios),
    }
    return scenarios, report
