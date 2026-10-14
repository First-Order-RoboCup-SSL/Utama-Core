"""Harvest `BenchScenario`s from a tagged tournament run, one per situation start.

A start is a situation of the goal ledger (`utama_core.analysis.goal_ledger`), from config_a's
(the candidate's) view, and carries its type (`ScenarioProvenance.situation`, one of
`goal_ledger.TYPES`):

- A restart, from the moment the referee awards it: `→ PREPARE_KICKOFF_*`,
  `→ PREPARE_PENALTY_*`, `STOP/BALL_PLACEMENT_* → DIRECT_FREE_*`, and `STOP → FORCE_START`.
  Played from there, the referee runs the restart as in the match (`apply_scenario`): the
  kickoff or penalty is prepared and goes live, the free kick is taken. Kickoffs and penalties
  were once anchored at the `NORMAL_START` that ends their preparation, which a bench runner
  could only replay as its own kickoff. Each start's perspective (candidate kicking or
  defending) is the colour the command names: restarts are asymmetric. A free kick's third is
  where the ball is, in the kicking side's attack frame, as the ledger's.
- A ball won in open play (`ball_won=True`): the start of each of the ledger's possession spells
  that is not a restart being taken, at the frame nearest it, if play is live there. Trigger
  `EVENT`; family `OPEN_PLAY_POSSESSION` when the candidate won the ball, `OPEN_PLAY_COUNTER`
  when the opponent did.

`harvest_run_dir` harvests every start of every trustworthy match, then keeps up to
`per_situation` of each type (`sample_by_situation`), so that rare situations are not
swamped by common ones; how often each happens is the bench's weights, not the bank's mix.

Match-level trust gate, per that same design pass: a match's replay is only
harvested from when its `<match_tag>.stats.json` (written by
`match.run_match`'s `stats_path`, see `MatchStats.to_json`) exists
and reports `stall_events == []`. A match with no stats file, or any stall
event, contributes nothing — this is the gate that would have caught the
925-file pre-fix contaminated replay run this session found (every one of
those matches was stuck at `PREPARE_KICKOFF_YELLOW` from t=0, so it either
has a `RESTART_STALL` recorded or, for even older runs with no watchdog at
all, no `.stats.json` to check — both fail closed here).

This module does NOT run a tournament itself — `harvest_run_dir` takes an
already-completed run directory (e.g. `replays/tournament_.../`) and is a
pure read; see `tools/scenario_bench.py` for the harvest → screen → score →
report pipeline this feeds.
"""

from __future__ import annotations

import collections
import functools
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utama_core.analysis.goal_ledger import TYPES, match_ledger, situation_type, third
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import Scenario, scenario_from_replay
from utama_core.scenario_bench.start import (
    BenchScenario,
    ScenarioFamily,
    ScenarioProvenance,
    ScenarioTrigger,
    is_near_duplicate,
    static_screen,
)

logger = logging.getLogger(__name__)

# Restart families keyed by the referee command that is transitioned INTO.
# `perspective` is candidate-relative and resolved per-match against which
# colour the candidate is playing — see `_perspective_family`.
_PENALTY_COMMANDS = frozenset({RefereeCommand.PREPARE_PENALTY_YELLOW, RefereeCommand.PREPARE_PENALTY_BLUE})
_KICKOFF_COMMANDS = frozenset({RefereeCommand.PREPARE_KICKOFF_YELLOW, RefereeCommand.PREPARE_KICKOFF_BLUE})
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
    """One restart awarded (see module docstring), from a `.intentions.jsonl`
    sidecar's referee-event rows."""

    sim_time: float
    prev_command: Optional[RefereeCommand]
    command: RefereeCommand
    # Which colour (True=yellow) takes the restart, as `command` names it; None for a
    # FORCE_START, which names no colour.
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
    if command in (_KICKOFF_COMMANDS | _PENALTY_COMMANDS | _DIRECT_FREE_COMMANDS):
        return command.name.endswith("_YELLOW")
    return None


def find_restart_transitions(sidecar_path: Path) -> list[RestartTransition]:
    """Scan a `.intentions.jsonl` sidecar for every restart awarded (see module docstring):
    `-> PREPARE_KICKOFF_*/PREPARE_PENALTY_*`, `STOP/BALL_PLACEMENT_* -> DIRECT_FREE_*`,
    `STOP -> FORCE_START`. A match's opening kickoff is its first row, after no command.

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

        is_restart = (
            (command in (_KICKOFF_COMMANDS | _PENALTY_COMMANDS) and prev_command != command)
            or (command in _DIRECT_FREE_COMMANDS and prev_command in _DIRECT_FREE_PREV_COMMANDS)
            or (command == RefereeCommand.FORCE_START and prev_command == RefereeCommand.STOP)
        )

        if is_restart and (ts - last_transition_ts) >= _MIN_RESTART_GAP_S:
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
    if transition.command in _KICKOFF_COMMANDS:
        return ScenarioFamily.KICKOFF
    if transition.command in _PENALTY_COMMANDS:
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
    if transition.kicking_is_yellow is None:
        return family, "candidate_kicking"
    if transition.kicking_is_yellow == candidate_is_yellow:
        return family, "candidate_kicking"
    if family == ScenarioFamily.DIRECT_FREE_ATTACKING:
        return ScenarioFamily.DIRECT_FREE_DEFENDING, "candidate_defending"
    return family, "candidate_defending"


def restart_situation(transition: RestartTransition, candidate_is_yellow: bool, scenario: Scenario) -> str:
    """A harvested restart's situation type from the candidate's view (`goal_ledger.TYPES`). The
    scenario is the candidate's view, defending the right goal: the candidate attacks -x."""
    if transition.command in _KICKOFF_COMMANDS:
        kind = "kickoff"
    elif transition.command in _PENALTY_COMMANDS:
        kind = "penalty"
    elif transition.command in _DIRECT_FREE_COMMANDS:
        kind = "free kick"
    else:
        kind = "force start"
    side = None
    if transition.kicking_is_yellow is not None:
        side = "friendly" if transition.kicking_is_yellow == candidate_is_yellow else "enemy"
    attack = -1.0 if side == "friendly" else 1.0
    return situation_type({"kind": kind, "side": side, "third": third(attack * scenario.ball_x)}, "friendly")


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


def _harvested(replay_path: Path, t: float, showing: Optional[RefereeCommand] = None) -> Optional[Scenario]:
    """The replay's field state at `t` (at the first frame showing `showing`, when given; see
    `scenario_from_replay`), or None if it can't be read or fails the static screen."""
    try:
        scenario = scenario_from_replay(replay_path, t, showing=showing)
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


def _slug(situation: str) -> str:
    return situation.replace(" third", "").replace(", ", "_").replace(" ", "_")


def harvest_replay(
    replay_path: Path,
    *,
    source_run_id: str,
    evaluator_version: str,
    candidate_is_yellow: bool = True,
    ball_won: bool = False,
) -> list[BenchScenario]:
    """Harvest every restart-transition `BenchScenario` from one replay and, with `ball_won`,
    every ball-won start (see module docstring).

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
        scenario = _harvested(replay_path, transition.sim_time, showing=transition.command)
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
            situation=restart_situation(transition, candidate_is_yellow, scenario),
        )
        scenarios.append(BenchScenario(scenario_id=scenario_id, scenario=scenario, provenance=provenance))

    # the ledger's friendly side is the replay's, config_a: the candidate
    if ball_won and replay_path.suffix == ".npz":
        for spell in match_ledger(replay_path)["spells"]:
            if spell["takes_restart"]:
                continue
            scenario = _harvested(replay_path, spell["t"])
            if scenario is None or scenario.referee_command not in _LIVE_PLAY_COMMANDS:
                continue
            own = spell["side"] == "friendly"
            situation = situation_type({**spell, "kind": "ball won"}, "friendly")
            provenance = ScenarioProvenance(
                source_run_id=source_run_id,
                evaluator_version=evaluator_version,
                trigger=ScenarioTrigger.EVENT,
                family=ScenarioFamily.OPEN_PLAY_POSSESSION if own else ScenarioFamily.OPEN_PLAY_COUNTER,
                anchor_tick=spell["t"],
                source_replay=replay_path,
                perspective="candidate_in_possession" if own else "candidate_defending",
                situation=situation,
            )
            scenario_id = f"{replay_path.stem}_t{spell['t']:.1f}_{_slug(situation)}"
            scenarios.append(BenchScenario(scenario_id=scenario_id, scenario=scenario, provenance=provenance))

    return scenarios


def sample_by_situation(scenarios: list[BenchScenario], per_situation: int, *, seed: int = 0) -> list[BenchScenario]:
    """Up to `per_situation` starts of each situation type, kickoffs half as many (their
    formations recur): drawn one match at a time in a seeded order, so a type's starts come
    from as many matches as it has, skipping any near-duplicate of a start already drawn."""
    rng = random.Random(seed)
    by_type: dict[str, dict[Path, list[BenchScenario]]] = collections.defaultdict(dict)
    for s in scenarios:
        by_type[s.provenance.situation].setdefault(s.provenance.source_replay, []).append(s)
    picked: list[BenchScenario] = []
    for typ in sorted(by_type, key=lambda t: TYPES.index(t) if t in TYPES else len(TYPES)):
        matches = [by_type[typ][m] for m in sorted(by_type[typ], key=str)]
        rng.shuffle(matches)
        for starts in matches:
            rng.shuffle(starts)
        cap = per_situation // 2 if typ.endswith("kickoff") else per_situation
        kept: list[BenchScenario] = []
        for round_ in range(max(len(m) for m in matches)):
            for starts in matches:
                if (
                    len(kept) < cap
                    and round_ < len(starts)
                    and not any(is_near_duplicate(starts[round_], k) for k in kept)
                ):
                    kept.append(starts[round_])
        picked += kept
    return picked


def _replay_files_in(run_dir: Path) -> list[Path]:
    # `<match>.sparse_referee.pkl` sits next to each replay; it is not a replay itself.
    pkls = [p for p in run_dir.glob("*.pkl") if not p.name.endswith(".sparse_referee.pkl")]
    return sorted({*run_dir.glob("*.npz"), *pkls})


def harvest_run_dir(
    run_dir: Path,
    *,
    evaluator_version: str,
    candidate_is_yellow: bool = True,
    per_situation: Optional[int] = None,
    seed: int = 0,
    workers: int = 1,
) -> tuple[list[BenchScenario], dict]:
    """Harvest every start (restarts and balls won) from every trustworthy match in `run_dir`
    (a completed `round_robin.py` run directory), then, with `per_situation`, keep a sample
    of each situation type (`sample_by_situation`).

    Returns `(scenarios, report)` where `report` is a small summary dict
    (`matches_seen`, `matches_trusted`, `matches_untrusted`,
    `scenarios_harvested`, and per situation type how many starts the run had and how
    many were kept) — a caller (the bench CLI) uses this to print "harvested N scenarios
    from M/K trustworthy matches" rather than only a bare list.
    """
    replay_paths = _replay_files_in(run_dir)
    trusted = [p for p in replay_paths if match_is_trustworthy(_stats_path_for(p))]
    for p in sorted(set(replay_paths) - set(trusted)):
        logger.info("Skipping untrusted match %s (missing/nonzero-stall .stats.json)", p.name)

    harvest = functools.partial(
        harvest_replay,
        source_run_id=run_dir.name,
        evaluator_version=evaluator_version,
        candidate_is_yellow=candidate_is_yellow,
        ball_won=True,
    )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_match = list(pool.map(harvest, trusted))
    else:
        per_match = [harvest(p) for p in trusted]
    scenarios = [s for match in per_match for s in match]
    found = collections.Counter(s.provenance.situation for s in scenarios)
    if per_situation is not None:
        scenarios = sample_by_situation(scenarios, per_situation, seed=seed)
    kept = collections.Counter(s.provenance.situation for s in scenarios)

    report = {
        "matches_seen": len(replay_paths),
        "matches_trusted": len(trusted),
        "matches_untrusted": len(replay_paths) - len(trusted),
        "scenarios_harvested": len(scenarios),
        "situations": {typ: {"found": found[typ], "kept": kept[typ]} for typ in TYPES if found[typ]},
    }
    return scenarios, report
