"""match.py — shared match-construction/running mechanics for the
evaluation driver (`evaluate.py`).

Why this exists: `evaluate.py`'s `run_match` and the since-retired
`full_match_tournament.py`'s `run_match_cell` were ~80% identical code (build
two strategies from their names, construct a referee with
the right kickoff-team/initial-command, wire up match_log/stats/replay paths,
build and step a `StrategyRunner`, read back the score) that had drifted into
two independent copies differing only in a few real parameters (duration,
whether side/kickoff are fixed or swept). Correctness fixes found in
one copy (the `PREPARE_KICKOFF_*` vs `FORCE_START` kickoff-ceremony fix; the
`_stats_to_dict` StallEvent-serialization fix) had to be manually ported to
the other and could easily have been missed. This module is the single place
that logic lives now; each script above is a thin CLI over it, differing only
in which strategies it selects, what duration it uses, and whether it sweeps
side/kickoff as independent axes. The strategies, referee and rsim runner of a
match come from `utama_core/run/sim_setup.py`, the one set-up the debugging tools share.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utama_core.bundle.strategy_file import SUFFIX, BundledStrategy
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_writer import ColumnarReplayWriterConfig
from utama_core.run.sim_setup import (  # noqa: F401 -- re-exported
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    load_side,
    sim_referee,
    sim_runner,
)
from utama_core.simulator.simulators.robosim.robosim_wrapper import enable_sim_reuse
from utama_core.simulator.utils.sim_noise import SimNoise
from utama_core.strategy import catalog
from utama_core.strategy.catalog import plain_name

TICKS_PER_SECOND = 60  # matches rsim's default step rate


def replay_name(run_dir: Path, match_tag: str) -> str:
    """A match's replay name: its path under REPLAY_BASE_PATH, no suffix, so the replay lands in
    `run_dir` as `<match_tag>.npz`."""
    return Path(os.path.relpath(Path(run_dir) / match_tag, REPLAY_BASE_PATH)).as_posix()


# default is excluded from the auto-discovered catalog:
# despite the name, it isn't a competitive team — it's the kernel's minimal
# single-tactic smoke-test scaffold (see its docstring), used across the test
# suite with as few as zero outfield robots, and plays a real match with 3 of 5
# outfield robots never issued a command. Every other strategy is playable here.
_CONFIG_NAMES = [name for name in catalog.names() if name != "default"]


def _short_name(config_name: str) -> str:
    if config_name.endswith(SUFFIX):
        return Path(config_name).stem
    return plain_name(config_name)


def _stats_to_dict(stats) -> dict:
    """`MatchStats.__dict__`, but with `stall_events` (a list of `StallEvent`
    dataclasses) turned into plain dicts so the result round-trips through
    `json.dump` in a run's `summary.json` — mirrors `MatchStats.to_json`'s own
    per-field shape rather than introducing a second serialization scheme.

    Found live, 2026-09-05: a full 24-match `full_match_tournament.py` run
    completed every match cleanly but crashed writing summary.json on the
    first stalled match's `StallEvent` (a bare `stats.__dict__` leaves
    `stall_events` as dataclass instances, which `json.dump` can't
    serialize), discarding the aggregate result entirely — the per-match
    `.stats.json` files, written earlier via `MatchStats.to_json`, were
    unaffected; only ad hoc summary aggregation hit this.
    """
    d = dict(stats.__dict__)
    d["stall_events"] = [
        {
            "kind": e.kind,
            "sim_time": e.sim_time,
            "tick": e.tick,
            "referee_command": e.referee_command,
            "duration_s": e.duration_s,
            "tactic_ids": list(e.tactic_ids),
            "robot_ids": list(e.robot_ids),
            "diagnosis": e.diagnosis,
        }
        for e in stats.stall_events
    ]
    d["fouls"] = [e.to_dict() for e in stats.fouls]
    return d


@dataclass
class MatchResult:
    """One match's outcome. `a_is_right`/`a_kicks_off` default to the
    `evaluate.py`'s fixed convention (config_a starts on the right and
    kicks off the first half; with halves the teams change ends at half-time).
    """

    config_a: str
    config_b: str
    score_a: int
    score_b: int
    a_is_right: bool = True
    a_kicks_off: bool = True
    stats: Optional[dict] = field(default=None, compare=False)

    @property
    def winner(self) -> str:
        if self.score_a > self.score_b:
            return self.config_a
        if self.score_b > self.score_a:
            return self.config_b
        return "draw"


def run_match(
    config_a_name: str,
    config_b_name: str,
    *,
    duration_seconds: float,
    a_is_right: bool = True,
    a_kicks_off: bool = True,
    run_dir: Optional[Path] = None,
    match_tag_suffix: str = "",
    sim_noise: Optional[SimNoise] = None,
) -> MatchResult:
    """Play one match between two strategies, until the referee calls full
    time (two halves of the profile's `half_duration_seconds` of playing time, the clock
    stopped whenever no team may play the ball) or `duration_seconds` of sim time pass,
    whichever is first. The cap only ends a match that can't finish: two 300 s halves took
    at most 870 s of sim time in `tournament_20261005_170958`'s replays.

    `config_a` is always yellow (a fixed convention — colour is never varied
    separately, since no tactic reads it and there's no rule reason to test
    it as its own axis); `a_is_right`/`a_kicks_off` are independent axes
    (side and kickoff are swept independently rather than blended into one
    "--both-sides" toggle: a live investigation found each one, isolated on
    its own, changes match outcomes on its own).

    A real `PREPARE_KICKOFF_YELLOW`/`_BLUE` ceremony (matching whichever side
    kicks off) is always used instead of `StrategyRunner`'s sim-mode default
    of `FORCE_START` — root-caused 2026-08-23 (see `docs/strategies.md`'s
    "Known open bugs"): `FORCE_START` releases both teams at the ball
    simultaneously from a mirror-symmetric formation, and sub-millimetre rsim
    physics noise then decides who's "closer to the ball" via
    `friendly_closer_to_ball`'s bare `<` comparison, cascading into a
    different match from a coin flip. A real kickoff ceremony (the prepare
    wait plus the kicker's walk to the centre circle) avoids that race.

    If `run_dir` is set, also records the full observability stack
    (structured intention log, aggregate stats, replay trail) under it — see
    `utama_core.engine.match_log`/`match_stats` and `utama_core.replay`.
    `match_tag_suffix` is appended to the per-match file tag (e.g. a side/
    kickoff cell tag like `_Rk`) so cells for the same pair don't collide.

    Both sides use the fpp planner (`tools/motion_planning_benchmark.py` compares planners).

    `sim_noise`, if set, plays the match with seeded latency, kick speed noise and vision noise
    (`utama_core/simulator/utils/sim_noise.py`, `docs/sim_realism.md`). The draws
    depend on the seed, team colour and robot id only, so the same `sim_noise` gives every
    pairing the same physics.
    """
    enable_sim_reuse()

    strategy_a = load_side(config_a_name)
    strategy_b = load_side(config_b_name)
    # config_a kicks off unless told otherwise; the referee's kickoff team and the ceremony follow it.
    referee = sim_referee(kickoff_team="yellow" if a_kicks_off else "blue")
    initial_command = RefereeCommand.PREPARE_KICKOFF_YELLOW if a_kicks_off else RefereeCommand.PREPARE_KICKOFF_BLUE

    match_tag = f"{_short_name(config_a_name)}_vs_{_short_name(config_b_name)}{match_tag_suffix}"
    extra_kwargs = {}
    if run_dir is not None:
        extra_kwargs["match_log_path"] = str(run_dir / f"{match_tag}.intentions.jsonl")
        extra_kwargs["stats_path"] = str(run_dir / f"{match_tag}.stats.json")
        # The writer puts a replay at REPLAY_BASE_PATH / replay_name: name it so the .npz lands in
        # run_dir beside the match's other files, however deep run_dir is (eval_<id>/round_<k>/).
        extra_kwargs["replay_writer_config"] = ColumnarReplayWriterConfig(
            replay_name=replay_name(run_dir, match_tag), overwrite_existing=True
        )

    runner, _ = sim_runner(
        strategy_a,
        strategy_b,
        sim_noise=sim_noise,
        initial_command=initial_command,
        referee=referee,
        my_team_is_right=a_is_right,
        control_scheme="fpp",
        **extra_kwargs,
    )

    try:
        for _ in range(int(duration_seconds * TICKS_PER_SECOND)):
            runner.step_once()
            if runner.my.game.referee is not None and runner.my.game.referee.stage == Stage.POST_GAME:
                break
        ref_data = runner.my.game.referee
        score_a = ref_data.yellow_team.score
        score_b = ref_data.blue_team.score
        stats = runner.match_stats.finalize() if runner.match_stats is not None else None
    finally:
        runner.close()
        for strategy in (strategy_a, strategy_b):
            if isinstance(strategy, BundledStrategy):
                strategy.close()

    return MatchResult(
        config_a=config_a_name,
        config_b=config_b_name,
        score_a=score_a,
        score_b=score_b,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
        stats=_stats_to_dict(stats) if stats is not None else None,
    )
