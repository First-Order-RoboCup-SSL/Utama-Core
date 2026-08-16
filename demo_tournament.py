"""demo_tournament.py — Round-robin every `kernel_strategy.py` config against every other.

Run:
    pixi run python demo_tournament.py

What this does
--------------
Headless rsim, no external process required. For each distinct pair of the
`build_*_kernel_strategy` factories in `utama_core.kernel.kernel_strategy`,
runs one `StrategyRunner` match (6v6: 1 goalkeeper + 5 outfield robots per
side — enough for every factory's minimum, including the `min_attack=2`
configs and `build_three_slot_kernel_strategy`'s three concurrent slots),
steps it for `MATCH_DURATION_SECONDS` of sim time, and records the final
score from `CustomReferee`'s scoreboard.

Deliberately the smallest mechanism that answers "how do these configs do
against each other": a for-loop over `StrategyRunner` matches and a plain
tally, not a new `Tournament`/`Runner` class, no persistence layer, no
bracketing/seeding. Round-robin (every pair once) rather than anything more
elaborate — there are only `C(n, 2)` pairs for the current catalog size, all
of which comfortably fit in one run. See `docs/roadmap.md`'s "Multi-strategy
/ tournament evaluation infra" section for why this stays this small: the
kernel's `Strategy`/`Tactic`/`Partitioner` primitives already solve the
"many things owning dynamic robot subsets" problem this would otherwise
need to reinvent — a tournament script is purely a driver on top of
`StrategyRunner`'s existing `opp_strategy` support, not a new mechanism.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from utama_core.custom_referee import CustomReferee
from utama_core.kernel import kernel_strategy
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

N_OUTFIELD = 5  # + 1 goalkeeper per side
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
MATCH_DURATION_SECONDS = 60.0
TICKS_PER_SECOND = 60  # matches rsim's default step rate

_CONFIG_NAMES = [
    name
    for name in dir(kernel_strategy)
    if name.startswith("build_") and name.endswith("_kernel_strategy") and callable(getattr(kernel_strategy, name))
]


@dataclass
class MatchResult:
    config_a: str
    config_b: str
    score_a: int
    score_b: int

    @property
    def winner(self) -> str:
        if self.score_a > self.score_b:
            return self.config_a
        if self.score_b > self.score_a:
            return self.config_b
        return "draw"


def run_match(config_a_name: str, config_b_name: str) -> MatchResult:
    build_a = getattr(kernel_strategy, config_a_name)
    build_b = getattr(kernel_strategy, config_b_name)

    strategy_a = AbstractStrategy(build_kernel_strategy=build_a(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_b(OUTFIELD_ROBOT_IDS))

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )

    runner = StrategyRunner(
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
    )

    try:
        for _ in range(int(MATCH_DURATION_SECONDS * TICKS_PER_SECOND)):
            runner.step_once()
        ref_data = runner.my.game.referee
        score_a = ref_data.yellow_team.score
        score_b = ref_data.blue_team.score
    finally:
        runner.close()

    return MatchResult(config_a=config_a_name, config_b=config_b_name, score_a=score_a, score_b=score_b)


def main() -> None:
    # Optional CLI args: config names (with or without the `build_`/
    # `_kernel_strategy` wrapping) to run instead of the full auto-discovered
    # catalog — useful for a quick check of one or two configs without
    # waiting on every pair, e.g. `python demo_tournament.py default
    # low_block`. `--sequential` forces the old one-process-at-a-time loop
    # (useful for debugging a specific match without pool noise); otherwise
    # matches run in a process pool since each `run_match` call is fully
    # self-contained (its own StrategyRunner, its own rSim subprocess, no
    # shared state with any other match) — this is the "many independent
    # matches" case, not the harder "parallelize robots within one match"
    # case (shared per-tick obstacle state, GIL-bound Python, would need its
    # own design). No args keeps today's behaviour: every config, once each.
    # `--max-workers N` caps the pool size, e.g. to leave headroom on a shared
    # machine — os.cpu_count() ignores CPU affinity/cgroup limits, so a
    # `taskset`-restricted run would otherwise still size the pool for all
    # cores and oversubscribe.
    args = sys.argv[1:]
    sequential = "--sequential" in args
    args = [a for a in args if a != "--sequential"]
    max_workers_override: int | None = None
    if "--max-workers" in args:
        idx = args.index("--max-workers")
        max_workers_override = int(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]

    if args:
        requested = set(args)
        config_names = [
            name
            for name in _CONFIG_NAMES
            if name in requested or name.removeprefix("build_").removesuffix("_kernel_strategy") in requested
        ]
        unmatched = requested - {
            n for name in config_names for n in (name, name.removeprefix("build_").removesuffix("_kernel_strategy"))
        }
        if unmatched:
            raise SystemExit(f"Unknown config name(s): {sorted(unmatched)}. Available: {sorted(_CONFIG_NAMES)}")
        if len(config_names) < 2:
            raise SystemExit("Need at least 2 configs to play a round-robin.")
    else:
        config_names = _CONFIG_NAMES

    pairs = list(itertools.combinations(sorted(config_names), 2))
    print(f"Round-robin: {len(config_names)} configs, {len(pairs)} matches")
    print(f"{N_OUTFIELD + 1}v{N_OUTFIELD + 1}, {MATCH_DURATION_SECONDS:.0f}s sim time per match, headless rsim")
    if not sequential:
        default_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = min(len(pairs), max_workers_override or default_workers)
        print(f"Running {n_workers} matches concurrently (--sequential to disable)\n")
    else:
        print("Running sequentially\n")

    results: list[MatchResult] = []
    wins: dict[str, int] = {name: 0 for name in config_names}
    draws: dict[str, int] = {name: 0 for name in config_names}

    def _record(result: MatchResult) -> None:
        results.append(result)
        if result.winner == "draw":
            draws[result.config_a] += 1
            draws[result.config_b] += 1
        else:
            wins[result.winner] += 1
        print(
            f"{result.config_a:<40} {result.score_a} - {result.score_b} "
            f"{result.config_b:<40} winner={result.winner}",
            flush=True,
        )

    if sequential:
        for config_a_name, config_b_name in pairs:
            _record(run_match(config_a_name, config_b_name))
    else:
        # Matches complete out of submission order under a process pool —
        # printed as they finish rather than buffered back into pair order,
        # since waiting to preserve order would throttle everything to the
        # slowest in-flight match. Final standings are still sorted, so the
        # only user-visible reordering is the interleaved progress log.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(run_match, a, b) for a, b in pairs]
            for future in as_completed(futures):
                _record(future.result())

    print("\nStandings (wins, draws):")
    for name in sorted(config_names, key=lambda n: (-wins[n], -draws[n])):
        print(f"  {name:<40} {wins[name]}W {draws[name]}D")


if __name__ == "__main__":
    main()
