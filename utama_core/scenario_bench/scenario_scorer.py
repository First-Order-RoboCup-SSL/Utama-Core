"""Score one `BenchScenario` for a candidate strategy against an opponent
strategy — the "outer loop, fast half" scorer from `docs/roadmap.md` item 14.

Builds a fresh headless `StrategyRunner` (same construction pattern
`repro_from_replay.py` uses), teleports it into the scenario's field state
via `apply_scenario`, ticks forward `horizon_s` sim seconds, and reads the
result off `MatchStats` — never inventing a new metric here, only reusing
the counters `utama_core.engine.match_stats` already accumulates live (see
that module's docstring for where each one came from).

Per-scenario outcome is a fixed ordinal scale, from the CANDIDATE's
perspective, per the design pass in item 14:
    goal_for > shot_on_target > entry_retained > neutral >
    turnover > shot_conceded > goal_against
plus independent `foul` and `stalled` flags (a stall is reported, never ranked:
it can be the strategy's fault or the planner's/referee's). TURNOVER counts real
ball losses only, the same rule as `turnover_breakdown`: the opponent kept the
ball for more than `FLICKER_S` (raw `MatchStats.turnovers` is mostly two robots
on one ball flipping "nearest"), or a restart was given to the opponent while we
had the ball. `score_scenario` returns the raw
`MatchStats` for both the candidate-as-friendly run so a caller (the bench
CLI) can compute paired differentials against a baseline strategy on the
same scenario+seed, not just look at one run in isolation — a single run's
absolute outcome is not the bench's unit of comparison (see item 14's
"always as differentials vs the opponent, never absolute").

Each start also measures `SIGNALS`, the candidate's side of the signals in
`docs/signals.md` over the scored window, from the same `ChanceTracker` the round-robin
analysis uses: an outcome says how a start ended, these say what happened in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from utama_core.analysis.chances import ChanceTracker
from utama_core.analysis.turnover_breakdown import ENEMY_RESTARTS, FLICKER_S, LIVE
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_stats import MatchStats
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_writer import ColumnarReplayWriterConfig
from utama_core.replay.scenario import apply_scenario
from utama_core.run import StrategyRunner
from utama_core.scenario_bench.start import BenchScenario
from utama_core.strategy import kernel_strategy

TICKS_PER_SECOND = 60  # matches round_robin.py/rsim's default step rate
N_OUTFIELD = 5
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))


class ScenarioOutcome(IntEnum):
    """Ordinal outcome scale, candidate's perspective. Higher is better for
    the candidate. `NEUTRAL` is the value when nothing decisive happened —
    the fragment simply ran out its horizon."""

    GOAL_AGAINST = -3
    SHOT_CONCEDED = -2
    TURNOVER = -1
    NEUTRAL = 0
    ENTRY_RETAINED = 1
    SHOT_ON_TARGET = 2
    GOAL_FOR = 3


@dataclass(frozen=True)
class ScenarioScoreResult:
    scenario_id: str
    outcome: ScenarioOutcome
    foul: bool
    horizon_s: float
    ticks_run: int
    stats: MatchStats
    error: Optional[str] = None
    stalled: bool = False
    signals: Optional[dict[str, float]] = None  # see `start_signals`


# Signal -> +1 where more is usually better for the candidate, -1 where less is.
SIGNALS = {
    "shots": 1,
    "open_shots": 1,  # shots weighted by the share of the goal mouth left open
    "regains_to_shot": 1,
    "entries": 1,
    "real_losses": -1,
    "shots_faced": -1,
    "open_shots_faced": -1,
    "danger_s": -1,  # seconds the opponent held the ball in the candidate's defensive third
}


def start_signals(chances: dict, *, real_losses: int, entries: int) -> dict[str, float]:
    """The candidate's ("friendly") `SIGNALS` over one start, from `ChanceTracker.result()`."""
    shots = [s for s in chances["shots"] if s["side"] == "friendly"]
    faced = [s for s in chances["shots"] if s["side"] == "enemy"]
    return {
        "shots": len(shots),
        "open_shots": round(sum(s["open_goal"] for s in shots), 2),
        "regains_to_shot": sum(r["shot_after_s"] is not None for r in chances["regains"] if r["side"] == "friendly"),
        "entries": entries,
        "real_losses": real_losses,
        "shots_faced": len(faced),
        "open_shots_faced": round(sum(s["open_goal"] for s in faced), 2),
        "danger_s": chances["danger"]["friendly"]["s"],
    }


def _resolve_config_name(name: str) -> str:
    if not name.startswith("build_"):
        name = f"build_{name}"
    if not name.endswith("_kernel_strategy"):
        name = f"{name}_kernel_strategy"
    if not hasattr(kernel_strategy, name):
        raise ValueError(f"Unknown strategy config {name!r}")
    return name


def _build_runner(
    candidate_config: str, opponent_config: str, *, stats_path: str, record: Optional[str] = None
) -> StrategyRunner:
    """`record`: a replay name relative to `REPLAY_BASE_PATH`; when set, the run writes
    `<record>.npz` (the replay) and `<record>.intentions.jsonl` (the match log) there."""
    build_candidate = getattr(kernel_strategy, _resolve_config_name(candidate_config))
    build_opponent = getattr(kernel_strategy, _resolve_config_name(opponent_config))
    strategy_a = AbstractStrategy(build_kernel_strategy=build_candidate(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_opponent(OUTFIELD_ROBOT_IDS))

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )

    return StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        referee_initial_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
        # The round-robins the banks are harvested from run fpp (tournament_lib.run_match);
        # a start played with another planner measures something those matches never did.
        control_scheme="fpp",
        stats_path=stats_path,
        **(
            {
                "replay_writer_config": ColumnarReplayWriterConfig(replay_name=record, overwrite_existing=True),
                "match_log_path": str(REPLAY_BASE_PATH / f"{record}.intentions.jsonl"),
            }
            if record is not None
            else {}
        ),
    )


class _RealLossWatch:
    """Counts the candidate's real ball losses tick by tick (see module docstring).
    Friendly is always yellow here, so the opponent's restarts are the BLUE ones."""

    def __init__(self) -> None:
        self.losses = 0
        self._pending_since: Optional[float] = None  # a live turnover not yet confirmed
        self._turnovers = 0
        self._cmd = None
        self._we_had_it = False

    def step(self, t: float, cmd, turnovers: int, poss_side: Optional[str]) -> None:
        if cmd in LIVE and turnovers > self._turnovers and self._pending_since is None:
            self._pending_since = t
        if self._pending_since is not None:
            if poss_side == "friendly":
                self._pending_since = None  # won back within the flicker window
            elif t - self._pending_since > FLICKER_S:
                self.losses += 1
                self._pending_since = None
        if cmd in ENEMY_RESTARTS and self._cmd not in ENEMY_RESTARTS and self._we_had_it:
            self.losses += 1
            self._pending_since = None  # the turnover that led here is this same loss
        self._turnovers, self._cmd, self._we_had_it = turnovers, cmd, poss_side == "friendly"


def _classify_outcome(
    before: MatchStats, after: MatchStats, own_goal: bool, conceded_goal: bool, real_losses: int = 0
) -> ScenarioOutcome:
    """Diff two `MatchStats` snapshots (candidate is always "friendly" —
    `apply_scenario`/`_build_runner` always construct the candidate as
    `runner.my`/yellow/right) into one ordinal `ScenarioOutcome`.

    Priority order matches the ordinal scale itself: a goal dominates a
    shot, a shot dominates an entry/turnover, ties fall to NEUTRAL. Only
    the deltas over the scenario's horizon matter, not the absolute counts
    (a scenario doesn't start at zero shots if harvested mid-match).
    """
    if own_goal:
        return ScenarioOutcome.GOAL_FOR
    if conceded_goal:
        return ScenarioOutcome.GOAL_AGAINST

    d_shots_friendly = after.shots.get("friendly", 0) - before.shots.get("friendly", 0)
    d_shots_enemy = after.shots.get("enemy", 0) - before.shots.get("enemy", 0)
    if d_shots_friendly > 0:
        return ScenarioOutcome.SHOT_ON_TARGET
    if d_shots_enemy > 0:
        return ScenarioOutcome.SHOT_CONCEDED

    d_entries = after.attacking_third_entries - before.attacking_third_entries
    if real_losses > 0:
        return ScenarioOutcome.TURNOVER
    if d_entries > 0:
        return ScenarioOutcome.ENTRY_RETAINED

    return ScenarioOutcome.NEUTRAL


def score_scenario(
    bench_scenario: BenchScenario,
    *,
    candidate_config: str,
    opponent_config: str,
    horizon_s: float = 20.0,
    stats_path: str = "/tmp/scenario_bench_stats.json",
    record: Optional[str] = None,
) -> ScenarioScoreResult:
    """Run `bench_scenario` for `candidate_config` (as the candidate, always
    friendly/yellow/right) against `opponent_config`, ticking forward
    `horizon_s` sim seconds, and return the ordinal outcome + full
    `MatchStats` diff.

    `bench_scenario.lead_in_s` (nonzero for event-triggered scenarios, see
    `start.py`'s docstring on the mem-loss lead-in) is ticked
    BEFORE the scored window starts, so both policies get a runway to
    reconstruct roles — the `MatchStats` "before" snapshot is taken after
    the lead-in, not at the raw teleport.
    """
    geometry = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
    scenario = bench_scenario.to_scenario()

    try:
        runner = _build_runner(candidate_config, opponent_config, stats_path=stats_path, record=record)
    except Exception as exc:  # noqa: BLE001 - report as a failed score, not a crash (e.g. unknown config name)
        return ScenarioScoreResult(
            scenario_id=bench_scenario.scenario_id,
            outcome=ScenarioOutcome.NEUTRAL,
            foul=False,
            horizon_s=horizon_s,
            ticks_run=0,
            stats=MatchStats({}, {}, {}),
            error=f"runner construction failed: {exc}",
        )

    try:
        apply_scenario(runner, scenario, verify=True)
    except Exception as exc:  # noqa: BLE001 - report as a failed score, not a crash
        runner.close()
        return ScenarioScoreResult(
            scenario_id=bench_scenario.scenario_id,
            outcome=ScenarioOutcome.NEUTRAL,
            foul=False,
            horizon_s=horizon_s,
            ticks_run=0,
            stats=runner.match_stats.finalize() if runner.match_stats is not None else MatchStats({}, {}, {}),
            error=f"apply_scenario failed: {exc}",
        )

    try:
        lead_in_ticks = int(bench_scenario.lead_in_s * TICKS_PER_SECOND)
        for _ in range(lead_in_ticks):
            runner.step_once()

        before = runner.match_stats.finalize() if runner.match_stats is not None else MatchStats({}, {}, {})

        own_goal = False
        conceded_goal = False
        acc = runner.match_stats
        loss_watch = _RealLossWatch()
        chance_tracker = ChanceTracker()
        # Regains count only after a restart into live play, not after the start itself:
        # an open-play start is already live, and isn't one.
        live_since = -math.inf
        prev_cmd = None
        horizon_ticks = int(horizon_s * TICKS_PER_SECOND)
        ticks_run = 0
        for _ in range(horizon_ticks):
            shots_before = dict(acc._shots) if acc is not None else None
            runner.step_once()
            ticks_run += 1
            frame = runner.my.current_game_frame
            if acc is not None:
                cmd = frame.referee.referee_command if frame.referee else None
                if cmd in LIVE and prev_cmd is not None and prev_cmd not in LIVE:
                    live_since = frame.ts
                prev_cmd = cmd
                loss_watch.step(ticks_run / TICKS_PER_SECOND, cmd, acc._turnovers, acc._poss_side)
                chance_tracker.step(frame, acc, cmd, live_since, shots_before)
            if frame.ball is not None:
                bx, by = frame.ball.p.x, frame.ball.p.y
                # Friendly is always right-defending (my_team_is_right=True; a 20 s
                # start never reaches half-time, where the teams change ends), so a ball in the RIGHT goal is conceded by friendly and a
                # ball in the LEFT goal is scored by friendly.
                if geometry.is_in_right_goal(bx, by):
                    conceded_goal = True
                    break
                if geometry.is_in_left_goal(bx, by):
                    own_goal = True
                    break

        after = runner.match_stats.finalize() if runner.match_stats is not None else before
        foul = sum(after.rule_event_counts.values()) > sum(before.rule_event_counts.values())
        outcome = _classify_outcome(before, after, own_goal, conceded_goal, loss_watch.losses)

        return ScenarioScoreResult(
            scenario_id=bench_scenario.scenario_id,
            outcome=outcome,
            foul=foul,
            horizon_s=horizon_s,
            ticks_run=ticks_run,
            stats=after,
            stalled=len(after.stall_events) > len(before.stall_events),
            signals=start_signals(
                chance_tracker.result(),
                real_losses=loss_watch.losses,
                entries=after.attacking_third_entries - before.attacking_third_entries,
            ),
        )
    finally:
        runner.close()
