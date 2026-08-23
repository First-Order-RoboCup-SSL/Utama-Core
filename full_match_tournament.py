"""Full-length, side/kickoff-decoupled round-robin among the `competitive`-tier
strategies from `docs/strategies.md` (`counter_flow`, `tiki_taka`, `zone_fluid`,
`counter_press`), at full-match duration (two 300s halves = 600s sim time,
matching `half_duration_seconds` in `docs/referee_integration.md`) instead of
`tournament.py`'s default 60s smoke-test length.

Why this exists instead of `tournament.py --both-sides`
--------------------------------------------------------
Every match here is fully deterministic — same tactic code, same fixed
formation generator, no seeded randomness in the sim itself (confirmed live:
re-running one fixture twice byte-for-byte reproduced the same score both
times). That means naively re-running the *same* fixture buys zero new
information, and `--both-sides` doesn't fix this cleanly either: swapping
which config is `config_a` simultaneously flips *three* things at once — which
physical side it starts on, its team colour, and (since `kickoff_team` is
pinned to "yellow" in the `simulation` referee profile and `config_a` is
always yellow) whether it kicks off. A live investigation this session found
that this combined swap can itself flip a match's outcome (`counter_flow` vs
`tiki_taka`: 0-2 one way, 1-0 the other), and that side and kickoff, isolated
individually via one-off `CustomReferee`/`StrategyRunner` construction, each
also changed the result on their own — so a 2-game "both sides" sample can't
actually attribute a result to any one cause, only to "the whole swap."

This module decouples side and kickoff into independent axes (colour is left
tied to "config_a is yellow" as a fixed convention — there's no rule reason to
vary it separately, and every tactic already reads `my_team_is_right`, never
raw colour). Per pair (A, B), this runs all 4 combinations of {A plays right,
A plays left} x {A kicks off, B kicks off} — 4 independent full-length
matches per pair, each attributable to a specific structural cell rather than
a blended "both sides" average.

No per-cell repeat sampling: an earlier version of this script added seeded
per-robot starting-formation jitter (`gauss(0, 0.15m)` position,
`gauss(0, 0.2rad)` orientation) as a third axis to get more independent
samples per cell. A 48-match run (`replays/tournament_20260822_234112/`,
written up in `docs/strategies.md`) showed every jittered-seed pair produced
*byte-identical* scores in all 4 side x kickoff cells across all 6 pairs —
that magnitude of formation noise never changed which branch a match took.
Removed rather than kept as dead weight; the real per-pair sample size here
is the 4 side x kickoff cells, not 4 x N seeds. If more independent samples
are needed later, the jitter magnitude would need to be much larger, or the
randomness would need to come from somewhere that actually changes early
tactical branching (e.g. randomizing initial possession), not sub-half-metre
formation noise.

Real kickoff ceremony, not simultaneous force-start
----------------------------------------------------
That same 48-match run also surfaced why "side" dominated every pair's
outcome: `StrategyRunner` defaults sim-mode matches to
`RefereeCommand.FORCE_START` (see its own comment — "sim matches... releas[e]
both teams at the ball at the same instant"), which meant every match to
date started with both teams' full pickers already live and racing for a
ball sitting exactly on the centre line, with mirror-symmetric formations.
rsim's physics does not resolve that mirror-symmetric setup with perfect
left/right symmetry (traced: ~0.1-0.3mm off a true mirror after 1 tick), and
`_friendly_closer_to_ball`'s bare `<` comparison (no tie margin) turns that
noise into a hard, match-shaping tactical branch (attack-heavy vs
press-heavy split) that both `counter_flow` and `tiki_taka`'s pickers commit
to immediately and never revisit. This module now seeds a real
`PREPARE_KICKOFF_YELLOW`/`_BLUE` (matching `kickoff_team`/`a_kicks_off`) via
`referee_initial_command` instead, so play only starts after the normal
`prepare_duration_seconds` wait plus the kicker actually walking to the
centre circle — verified via direct trace to remove the tick-1 coin flip
(the edge now agrees between mirrored sides through the whole approach
phase). This alone does *not* fully eliminate the underlying tie — first
ball touch is still a near-zero-distance moment either way, so the same
sub-millimetre rsim noise can still flip `_friendly_closer_to_ball` right at
contact. `kernel_strategy.py`'s `_CLOSER_TO_BALL_MARGIN = 0.05` (added the
same session) is the second half of the fix — a real hysteresis margin on
`_friendly_closer_to_ball` itself, not just the kickoff-ceremony timing —
and together both are a real, verified improvement over a simultaneous
release, though not a total elimination of the tie: two independent physics
runs converging on a moving threshold can still cross a fixed margin
boundary on different ticks even with a real (if tiny) kinematic difference
between them. See `docs/strategies.md`'s "Known open bugs" for the full
trace evidence on both fixes. Not yet done: re-running this module's 4-cell
decoupled tournament with both fixes active to measure how much the
side-dependence pattern shrinks in aggregate.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tournament
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.replay_writer import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

tournament.MATCH_DURATION_SECONDS = 600.0  # full match: two 300s halves

COMPETITIVE = [
    "build_counter_flow_kernel_strategy",
    "build_tiki_taka_kernel_strategy",
    "build_zone_fluid_kernel_strategy",
    "build_counter_press_kernel_strategy",
]

N_OUTFIELD = tournament.N_OUTFIELD
OUTFIELD_ROBOT_IDS = tournament.OUTFIELD_ROBOT_IDS


@dataclass
class CellResult:
    config_a: str
    config_b: str
    a_is_right: bool
    a_kicks_off: bool
    score_a: int
    score_b: int
    stats: Optional[dict] = field(default=None, compare=False)

    @property
    def winner(self) -> str:
        if self.score_a > self.score_b:
            return self.config_a
        if self.score_b > self.score_a:
            return self.config_b
        return "draw"


def run_match_cell(
    config_a_name: str,
    config_b_name: str,
    a_is_right: bool,
    a_kicks_off: bool,
    run_dir: Optional[Path] = None,
) -> CellResult:
    """Play one full-length match with side and kickoff set explicitly,
    independent of each other. `config_a` is always yellow (a fixed
    convention — colour is never varied separately, since no tactic reads it
    and there's no rule reason to test it as its own axis).
    """
    build_a = getattr(kernel_strategy, config_a_name)
    build_b = getattr(kernel_strategy, config_b_name)

    strategy_a = AbstractStrategy(build_kernel_strategy=build_a(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_b(OUTFIELD_ROBOT_IDS))

    profile = load_profile("simulation")
    kickoff_team = "yellow" if a_kicks_off else "blue"
    profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, kickoff_team=kickoff_team))
    referee = CustomReferee(profile, n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1)
    initial_command = RefereeCommand.PREPARE_KICKOFF_YELLOW if a_kicks_off else RefereeCommand.PREPARE_KICKOFF_BLUE

    side_tag = "R" if a_is_right else "L"
    kickoff_tag = "K" if a_kicks_off else "k"
    match_tag = (
        f"{tournament._short_name(config_a_name)}_vs_{tournament._short_name(config_b_name)}"
        f"_{side_tag}{kickoff_tag}"
    )
    extra_kwargs = {}
    if run_dir is not None:
        extra_kwargs["match_log_path"] = str(run_dir / f"{match_tag}.intentions.jsonl")
        extra_kwargs["stats_path"] = str(run_dir / f"{match_tag}.stats.json")
        extra_kwargs["replay_writer_config"] = ReplayWriterConfig(
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
        **extra_kwargs,
    )
    try:
        for _ in range(int(tournament.MATCH_DURATION_SECONDS * tournament.TICKS_PER_SECOND)):
            runner.step_once()
        ref_data = runner.my.game.referee
        score_a = ref_data.yellow_team.score
        score_b = ref_data.blue_team.score
        stats = runner.match_stats.finalize() if runner.match_stats is not None else None
    finally:
        runner.close()

    return CellResult(
        config_a=config_a_name,
        config_b=config_b_name,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
        score_a=score_a,
        score_b=score_b,
        stats=stats.__dict__ if stats is not None else None,
    )


def main() -> None:
    missing = [n for n in COMPETITIVE if n not in tournament._CONFIG_NAMES]
    if missing:
        raise SystemExit(f"Missing expected competitive config(s): {missing}")

    base_pairs = list(itertools.combinations(sorted(COMPETITIVE), 2))
    cells = list(itertools.product([True, False], [True, False]))
    jobs = [(a, b, a_is_right, a_kicks_off) for a, b in base_pairs for a_is_right, a_kicks_off in cells]

    run_id = datetime.now(timezone.utc).strftime("tournament_%Y%m%d_%H%M%S")
    run_dir = REPLAY_BASE_PATH / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Competitive-only decoupled round-robin: {len(COMPETITIVE)} configs, {len(base_pairs)} pairs")
    print(f"{len(cells)} cells/pair (side x kickoff) = {len(jobs)} matches")
    print(f"6v6, {tournament.MATCH_DURATION_SECONDS:.0f}s sim time per match (full match), headless rsim")
    print(f"Recording to replays/{run_id}/\n")

    # Each match is 1 pool-worker process + 2 robosim subprocesses (friendly +
    # enemy sim), so oversubscription hits at ~1/3 of cpu_count() concurrent
    # matches, not cpu_count() itself -- learned live: 15 workers on 16 cores
    # (45+ total processes) left every worker around 70% CPU with zero
    # matches finishing in 7+ minutes.
    n_workers = min(len(jobs), max(1, (os.cpu_count() or 1) // 3))
    print(f"Running {n_workers} matches concurrently\n")

    results: list[CellResult] = []
    wins = {name: 0 for name in COMPETITIVE}
    draws = {name: 0 for name in COMPETITIVE}

    def _record(result: CellResult) -> None:
        results.append(result)
        if result.winner == "draw":
            draws[result.config_a] += 1
            draws[result.config_b] += 1
        else:
            wins[result.winner] += 1
        side_tag = "right" if result.a_is_right else "left"
        kickoff_tag = "kickoff" if result.a_kicks_off else "no-kickoff"
        print(
            f"{result.config_a:<40} ({side_tag},{kickoff_tag}) "
            f"{result.score_a} - {result.score_b} {result.config_b:<40} winner={result.winner}",
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_match_cell, a, b, right, kickoff, run_dir) for a, b, right, kickoff in jobs]
        for future in as_completed(futures):
            _record(future.result())

    print("\nStandings (wins, draws) across all cells:")
    for name in sorted(COMPETITIVE, key=lambda n: (-wins[n], -draws[n])):
        print(f"  {name:<40} {wins[name]}W {draws[name]}D")

    summary = {
        "run_id": run_id,
        "config_names": sorted(COMPETITIVE),
        "match_duration_seconds": tournament.MATCH_DURATION_SECONDS,
        "results": [
            {
                "config_a": r.config_a,
                "config_b": r.config_b,
                "a_is_right": r.a_is_right,
                "a_kicks_off": r.a_kicks_off,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "winner": r.winner,
                "stats": r.stats,
            }
            for r in results
        ],
        "standings": {name: {"wins": wins[name], "draws": draws[name]} for name in COMPETITIVE},
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results + stats: replays/{run_id}/summary.json")


if __name__ == "__main__":
    main()
