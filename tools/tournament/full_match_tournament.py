"""Full-length, side/kickoff-decoupled round-robin among the `competitive`-tier
strategies from `docs/strategies.md` (`counter_flow`, `tiki_taka`, `zone_fluid`,
`counter_press`), two halves of 300 s of playing time per match like `round_robin.py`.
Each pair plays starting on both sides and with both first-half kickoffs (the teams change
ends at half-time, so each starting side is still a different match).

Match construction (build strategies, referee, StrategyRunner, kickoff
ceremony, run_dir file layout) lives in `tournament_lib.py`, shared with
`round_robin.py` — see that module's own docstring for why.

Why this exists instead of `round_robin.py --both-sides`
---------------------------------------------------------------
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
`friendly_closer_to_ball`'s bare `<` comparison (no tie margin) turns that
noise into a hard, match-shaping tactical branch (attack-heavy vs
press-heavy split) that both `counter_flow` and `tiki_taka`'s pickers commit
to immediately and never revisit. This module (via `tournament_lib.run_match`)
seeds a real `PREPARE_KICKOFF_YELLOW`/`_BLUE` (matching `a_kicks_off`)
instead, so play only starts after the normal `prepare_duration_seconds` wait
plus the kicker actually walking to the centre circle — verified via direct
trace to remove the tick-1 coin flip (the edge now agrees between mirrored
sides through the whole approach phase). This alone does *not* fully
eliminate the underlying tie — first ball touch is still a near-zero-distance
moment either way, so the same sub-millimetre rsim noise can still flip
`friendly_closer_to_ball` right at contact. `strategy/pickers.py`'s
`CLOSER_TO_BALL_MARGIN = 0.05` (added the same session) is the second half
of the fix — a real hysteresis margin on `friendly_closer_to_ball` itself,
not just the kickoff-ceremony timing — and together both are a real, verified
improvement over a simultaneous release, though not a total elimination of
the tie: two independent physics runs converging on a moving threshold can
still cross a fixed margin boundary on different ticks even with a real (if
tiny) kinematic difference between them. See `docs/strategies.md`'s "Known
open bugs" for the full trace evidence on both fixes.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tournament.tournament_lib import (  # noqa: F401 -- re-exported for existing callers
    _CONFIG_NAMES,
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    _short_name,
)
from tools.tournament.tournament_lib import run_match as _lib_run_match
from utama_core.config.settings import REPLAY_BASE_PATH

MAX_MATCH_SECONDS = 1200.0  # caps a match that can't reach full time; see round_robin.py

COMPETITIVE = [
    "build_counter_flow_kernel_strategy",
    "build_tiki_taka_kernel_strategy",
    "build_zone_fluid_kernel_strategy",
    "build_counter_press_kernel_strategy",
    "build_tiki_taka_plus_kernel_strategy",
]


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
    and there's no rule reason to test it as its own axis). Thin wrapper over
    `tournament_lib.run_match`; see its docstring for the match-construction
    details (kickoff ceremony, file layout, etc).
    """
    side_tag = "R" if a_is_right else "L"
    kickoff_tag = "K" if a_kicks_off else "k"
    result = _lib_run_match(
        config_a_name,
        config_b_name,
        duration_seconds=MAX_MATCH_SECONDS,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
        run_dir=run_dir,
        match_tag_suffix=f"_{side_tag}{kickoff_tag}",
    )
    return CellResult(
        config_a=result.config_a,
        config_b=result.config_b,
        a_is_right=result.a_is_right,
        a_kicks_off=result.a_kicks_off,
        score_a=result.score_a,
        score_b=result.score_b,
        stats=result.stats,
    )


def main() -> None:
    missing = [n for n in COMPETITIVE if n not in _CONFIG_NAMES]
    if missing:
        raise SystemExit(f"Missing expected competitive config(s): {missing}")

    # --no-save: skip the replay/intention-log/stats/summary.json trail for a
    # throwaway smoke-test run that doesn't need to be analyzed afterward —
    # this run's per-match replays can be tens to hundreds of MB each, and an
    # always-on run_dir with no opt-out is exactly how replays/ silently
    # filled up with unreferenced gigabytes before (cleaned up 2026-08-26;
    # see docs/STRATEGY_DEVELOPMENT.md's Observability section).
    no_save = "--no-save" in sys.argv[1:]

    base_pairs = list(itertools.combinations(sorted(COMPETITIVE), 2))
    cells = list(itertools.product([True, False], [True, False]))
    jobs = [(a, b, a_is_right, a_kicks_off) for a, b in base_pairs for a_is_right, a_kicks_off in cells]

    run_id = datetime.now(timezone.utc).strftime("tournament_%Y%m%d_%H%M%S")
    run_dir: Optional[Path] = None
    if not no_save:
        run_dir = REPLAY_BASE_PATH / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Competitive-only decoupled round-robin: {len(COMPETITIVE)} configs, {len(base_pairs)} pairs")
    print(f"{len(cells)} cells/pair (side x kickoff) = {len(jobs)} matches")
    print("6v6, two 300 s halves of playing time per match, headless rsim")
    if no_save:
        print("--no-save: not recording replay/intention-log/stats for this run\n")
    else:
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

    failures: list[BaseException] = []
    future_to_job = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_match_cell, a, b, right, kickoff, run_dir) for a, b, right, kickoff in jobs]
        future_to_job = dict(zip(futures, jobs))
        for future in as_completed(futures):
            try:
                _record(future.result())
            except Exception as exc:
                # A worker can die after it has already written its
                # stats/intentions/replay files to disk (e.g. a transient
                # empty/malformed read off the rsim subprocess's stdout pipe
                # on shutdown) -- that's a lost CellResult, not lost match
                # data. Previously this propagated straight out of
                # as_completed() and killed the whole run, discarding every
                # other already-completed result's summary.json entry along
                # with it (found live, 2026-09-01: all 40/40 match cells had
                # written their files fine, but one bad future.result() took
                # down collection before any of them could be recorded).
                a, b, right, kickoff = future_to_job[future]
                print(f"ERROR: match cell {a} vs {b} (right={right}, kickoff={kickoff}) failed: {exc!r}", flush=True)
                failures.append(exc)

    if failures:
        print(f"\n{len(failures)} of {len(jobs)} match cell(s) failed to report a result (see errors above).")

    print("\nStandings (wins, draws) across all cells:")
    for name in sorted(COMPETITIVE, key=lambda n: (-wins[n], -draws[n])):
        print(f"  {name:<40} {wins[name]}W {draws[name]}D")

    summary = {
        "run_id": run_id,
        "config_names": sorted(COMPETITIVE),
        "max_match_seconds": MAX_MATCH_SECONDS,
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
    if run_dir is not None:
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results + stats: replays/{run_id}/summary.json")


if __name__ == "__main__":
    main()
