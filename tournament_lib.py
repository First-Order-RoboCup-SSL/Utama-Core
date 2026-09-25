"""tournament_lib.py — shared match-construction/running mechanics for the
top-level tournament drivers (`smoke_tournament.py`, `full_match_tournament.py`,
`arena_tournament.py`).

Why this exists: `smoke_tournament.py`'s `run_match` and
`full_match_tournament.py`'s `run_match_cell` were ~80% identical code (build
two strategies from kernel_strategy factory names, construct a referee with
the right kickoff-team/initial-command, wire up match_log/stats/replay paths,
build and step a `StrategyRunner`, read back the score) that had drifted into
two independent copies differing only in a few real parameters (duration,
whether side/kickoff are fixed or swept, fuzzing). Correctness fixes found in
one copy (the `PREPARE_KICKOFF_*` vs `FORCE_START` kickoff-ceremony fix; the
`_stats_to_dict` StallEvent-serialization fix) had to be manually ported to
the other and could easily have been missed. This module is the single place
that logic lives now; each script above is a thin CLI over it, differing only
in which strategies it selects, what duration it uses, and whether it sweeps
side/kickoff as independent axes.

`arena_tournament.py` is NOT a thin wrapper over `run_match` here — it needs
per-tick instrumentation (target recorders, slot-state dumps) that `run_match`
does not and should not provide, since every other caller pays no cost for
it. It still shares `N_OUTFIELD`/`OUTFIELD_ROBOT_IDS`/`TICKS_PER_SECOND`/
`_CONFIG_NAMES`/`_short_name` from here rather than keeping its own copies.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.custom_referee.restart_fuzzer import RestartFuzzingReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_writer import ColumnarReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_OUTFIELD = 5  # + 1 goalkeeper per side
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
TICKS_PER_SECOND = 60  # matches rsim's default step rate

# build_default_kernel_strategy is excluded from the auto-discovered catalog:
# despite the name, it isn't a competitive team — it's the kernel's minimal
# single-tactic smoke-test scaffold (see its docstring), used across the test
# suite with as few as zero outfield robots, and the arena-strategy stats
# investigation confirmed it plays a real match with 3 of 5 outfield robots
# never issued a command ("zombie" robots, 0.0 motion all match). Fixing that
# would still only produce a deliberately-minimal team, not a useful
# comparison point. build_tiki_taka_kernel_strategy — the strongest, most
# complete team by the same stats investigation (live-state posture,
# possession-backed wins, no losses) — is the de facto baseline other configs
# get judged against instead; it needs no special-casing here since it's
# already just another entry in the catalog.
_CONFIG_NAMES = [
    name
    for name in dir(kernel_strategy)
    if name.startswith("build_")
    and name.endswith("_kernel_strategy")
    and callable(getattr(kernel_strategy, name))
    and name != "build_default_kernel_strategy"
]


def _short_name(config_name: str) -> str:
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


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
        }
        for e in stats.stall_events
    ]
    return d


@dataclass
class MatchResult:
    """One match's outcome. `a_is_right`/`a_kicks_off` default to the
    historical `smoke_tournament.py` fixed convention (config_a always right,
    always kicks off) — `full_match_tournament.py`'s decoupled sweep passes
    both explicitly per cell.
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
    control_scheme: str = "fpp",
    fuzz_seed: Optional[int] = None,
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0),
    factory_a: Optional[Callable] = None,
    factory_b: Optional[Callable] = None,
    render: bool = False,
) -> MatchResult:
    """Play one match between two kernel-strategy factories.

    `config_a` is always yellow (a fixed convention — colour is never varied
    separately, since no tactic reads it and there's no rule reason to test
    it as its own axis); `a_is_right`/`a_kicks_off` are independent axes
    (see `full_match_tournament.py`'s module docstring for why side and
    kickoff must be swept independently rather than blended into one
    "--both-sides" toggle: a live investigation found each one, isolated on
    its own, changes match outcomes on its own).

    A real `PREPARE_KICKOFF_YELLOW`/`_BLUE` ceremony (matching whichever side
    kicks off) is always used instead of `StrategyRunner`'s sim-mode default
    of `FORCE_START` — root-caused 2026-08-23 (see `docs/strategies.md`'s
    "Known open bugs"): `FORCE_START` releases both teams at the ball
    simultaneously from a mirror-symmetric formation, and sub-millimetre rsim
    physics noise then decides who's "closer to the ball" via
    `_friendly_closer_to_ball`'s bare `<` comparison, cascading into a
    different match from a coin flip. A real kickoff ceremony (the prepare
    wait plus the kicker's walk to the centre circle) avoids that race.

    If `run_dir` is set, also records the full observability stack
    (structured intention log, aggregate stats, replay trail) under it — see
    `utama_core.engine.match_log`/`match_stats` and `utama_core.replay`.
    `match_tag_suffix` is appended to the per-match file tag (e.g. a side/
    kickoff cell tag like `_Rk`) so cells for the same pair don't collide.

    `control_scheme` is used for both sides (matching StrategyRunner's
    default of falling back to `control_scheme` when `opp_control_scheme` is
    unset) — these tournament drivers compare strategies against each other,
    not motion planners against each other (see
    `tools/motion_planning_benchmark.py` for that).

    `fuzz_seed`, if set, swaps in `RestartFuzzingReferee` (see
    `utama_core/custom_referee/restart_fuzzer.py`) instead of plain
    `CustomReferee`, so this match's referee injects extra seeded-random
    legal restarts during live play.

    `factory_a`/`factory_b`, if set, replace the `kernel_strategy` lookup for
    that side — any `outfield_robot_ids -> build_kernel_strategy` callable
    (e.g. `build_openjev_kernel_strategy` with its options bound), which lets
    teams outside the auto-discovered catalog play; the config names are then
    only used as labels. `render` opens the live rsim window while stepping.
    """
    build_a = factory_a or getattr(kernel_strategy, config_a_name)
    build_b = factory_b or getattr(kernel_strategy, config_b_name)

    strategy_a = AbstractStrategy(build_kernel_strategy=build_a(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_b(OUTFIELD_ROBOT_IDS))

    # "simulation" profile's kickoff_team defaults to "yellow"; override it to
    # "blue" when B kicks off. Neither CustomReferee.from_profile_name nor
    # RestartFuzzingReferee.from_profile_name takes a kickoff_team override,
    # so build the profile by hand the same way full_match_tournament.py's
    # run_match_cell always did, then construct the referee from it directly.
    profile = load_profile("simulation")
    if not a_kicks_off:
        profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, kickoff_team="blue"))
    if fuzz_seed is not None:
        referee = RestartFuzzingReferee(
            profile,
            seed=fuzz_seed,
            interval_s=fuzz_interval_s,
            n_robots_yellow=N_OUTFIELD + 1,
            n_robots_blue=N_OUTFIELD + 1,
        )
    else:
        referee = CustomReferee(profile, n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1)
    initial_command = RefereeCommand.PREPARE_KICKOFF_YELLOW if a_kicks_off else RefereeCommand.PREPARE_KICKOFF_BLUE

    match_tag = f"{_short_name(config_a_name)}_vs_{_short_name(config_b_name)}{match_tag_suffix}"
    extra_kwargs = {}
    if run_dir is not None:
        extra_kwargs["match_log_path"] = str(run_dir / f"{match_tag}.intentions.jsonl")
        extra_kwargs["stats_path"] = str(run_dir / f"{match_tag}.stats.json")
        # replay_name is relative to REPLAY_BASE_PATH, not run_dir, since replays
        # live under a fixed replays/ root — nest it under the same tournament
        # subdirectory so the two stay next to each other on disk.
        #
        # Columnar (.npz) rather than pickle (.pkl): ~13 MB/match in the old
        # format vs. a fraction of that here, and every reader a tournament
        # run's replays are actually fed through — `load_frames_in_range`
        # (`replay_player.py`, used by `render_window`) and
        # `find_stuck_windows` (`stuck_detector.py`) — already dispatches on
        # `.npz` vs `.pkl` by extension, so nothing downstream of a
        # tournament run breaks. Only the interactive `play_replay`/
        # `get_latest_replay_name` CLI helpers in `replay_player.py` still
        # hardcode `.pkl`.
        extra_kwargs["replay_writer_config"] = ColumnarReplayWriterConfig(
            replay_name=f"{run_dir.name}/{match_tag}", overwrite_existing=True
        )

    runner = StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=a_is_right,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        referee_initial_command=initial_command,
        control_scheme=control_scheme,
        **extra_kwargs,
    )

    if render and runner.rsim_env is not None:
        runner.rsim_env.render_mode = "human"  # live pygame window, same switch StrategyRunner.run() flips
    try:
        for _ in range(int(duration_seconds * TICKS_PER_SECOND)):
            runner.step_once()
        ref_data = runner.my.game.referee
        score_a = ref_data.yellow_team.score
        score_b = ref_data.blue_team.score
        stats = runner.match_stats.finalize() if runner.match_stats is not None else None
    finally:
        runner.close()

    return MatchResult(
        config_a=config_a_name,
        config_b=config_b_name,
        score_a=score_a,
        score_b=score_b,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
        stats=_stats_to_dict(stats) if stats is not None else None,
    )
