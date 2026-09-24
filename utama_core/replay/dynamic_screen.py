"""Dynamic (play-forward) validity screen for bench scenarios.

Complements `bench_scenario.static_screen` (a single-snapshot check) with
the screen that actually catches degenerate scenarios, per the design pass
in `docs/roadmap.md` item 14 (2026-09-04): play the scenario forward with a
few policies across a few seeds, then classify:

    - DEAD: no shot/turnover/goal/foul in any run — zero discriminative
      power, drop.
    - DETERMINED: identical outcome regardless of seed and policy — nearly
      zero information, drop (or keep exactly one per family as a sanity
      anchor — that decision belongs to whoever curates the bank, not this
      module).
    - NOISY: outcome variance across seeds for the same policy is high —
      keep only if the family's noise floor can absorb it.
    - INFORMATIVE: outcome varies more with policy than with seed — keep.

rsim is deterministic given identical inputs, so a "seed" here is a
`bench_scenario.jittered` start: the same scenario with the robots away from
the ball nudged a few centimetres. Seed spread is the spread within one
opponent across seeds; policy spread is the spread of the per-opponent means.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from utama_core.replay.bench_scenario import BenchScenario, jittered
from utama_core.replay.scenario_scorer import ScenarioOutcome, score_scenario

# Default pool of policies to play a candidate scenario forward against for
# the dynamic screen — deliberately small and diverse (not the full
# tournament catalog), matching item 14's "champion vs self, and two or
# three pool members" guidance.
DEFAULT_SCREEN_POOL = ("build_low_block_kernel_strategy", "build_high_press_kernel_strategy")


class ScreenVerdict(Enum):
    DEAD = "dead"
    DETERMINED = "determined"
    NOISY = "noisy"
    INFORMATIVE = "informative"


@dataclass(frozen=True)
class DynamicScreenResult:
    scenario_id: str
    verdict: ScreenVerdict
    outcomes: tuple[int, ...]  # ScenarioOutcome values, one per (opponent, seed) run, opponent-major
    outcome_stdev: float
    any_decisive_event: bool  # True if any run had a non-NEUTRAL outcome or a foul
    seed_stdev: float = 0.0  # mean over opponents of the stdev across seeds
    policy_stdev: float = 0.0  # stdev of the per-opponent mean outcomes


# A run is "decisive" if the outcome differs from NEUTRAL or a foul fired —
# matches item 14's "shot, turnover, goal, or foul in any run" DEAD test.
_NEUTRAL = int(ScenarioOutcome.NEUTRAL)

# Outcome stdev at or below this (in ordinal units) across all runs counts
# as DETERMINED — every run landed on essentially the same result.
_DETERMINED_STDEV = 0.01


def screen_scenario(
    bench_scenario: BenchScenario,
    *,
    champion_config: str,
    pool_configs: Sequence[str] = DEFAULT_SCREEN_POOL,
    horizon_s: float = 20.0,
    repeats: int = 3,
) -> DynamicScreenResult:
    """Play `bench_scenario` forward with the champion against itself and each
    of `pool_configs` (duplicates dropped), `repeats` jittered starts each, and
    classify the outcome spread. NOISY when the seed spread is at least the
    policy spread: the outcome says more about the start than about who played."""
    opponents = tuple(dict.fromkeys((champion_config, *pool_configs)))
    per_opponent: list[list[int]] = []
    any_decisive = False

    for opponent_config in opponents:
        runs = []
        for seed in range(repeats):
            result = score_scenario(
                jittered(bench_scenario, seed),
                candidate_config=champion_config,
                opponent_config=opponent_config,
                horizon_s=horizon_s,
            )
            runs.append(int(result.outcome))
            if result.outcome != ScenarioOutcome.NEUTRAL or result.foul:
                any_decisive = True
        per_opponent.append(runs)

    outcomes = [o for runs in per_opponent for o in runs]
    stdev = statistics.pstdev(outcomes) if len(outcomes) > 1 else 0.0
    seed_stdev = statistics.fmean(statistics.pstdev(runs) if len(runs) > 1 else 0.0 for runs in per_opponent)
    means = [statistics.fmean(runs) for runs in per_opponent]
    policy_stdev = statistics.pstdev(means) if len(means) > 1 else 0.0

    if not any_decisive:
        verdict = ScreenVerdict.DEAD
    elif stdev <= _DETERMINED_STDEV:
        verdict = ScreenVerdict.DETERMINED
    elif seed_stdev >= policy_stdev:
        verdict = ScreenVerdict.NOISY
    else:
        verdict = ScreenVerdict.INFORMATIVE

    return DynamicScreenResult(
        scenario_id=bench_scenario.scenario_id,
        verdict=verdict,
        outcomes=tuple(outcomes),
        outcome_stdev=stdev,
        any_decisive_event=any_decisive,
        seed_stdev=seed_stdev,
        policy_stdev=policy_stdev,
    )
