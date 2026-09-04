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

rsim is deterministic per the same seed/inputs (see `tools/
metric_correlation.py`'s note that re-running a matchup adds no
information under a fixed seed) — so "several seeds" here means several
*policy pairings*, not RNG seeds in the traditional sense: this module
varies the opponent config across a small pool to get outcome spread,
since `score_scenario` itself has no RNG knob. If a future controller
gains actual stochasticity, this is the module to add a seed parameter to.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from utama_core.replay.bench_scenario import BenchScenario
from utama_core.replay.scenario_scorer import ScenarioOutcome, score_scenario

# Default pool of policies to play a candidate scenario forward against for
# the dynamic screen — deliberately small and diverse (not the full
# tournament catalog), matching item 14's "champion vs self, and two or
# three pool members" guidance.
DEFAULT_SCREEN_POOL = ("build_default_kernel_strategy",)


class ScreenVerdict(Enum):
    DEAD = "dead"
    DETERMINED = "determined"
    NOISY = "noisy"
    INFORMATIVE = "informative"


@dataclass(frozen=True)
class DynamicScreenResult:
    scenario_id: str
    verdict: ScreenVerdict
    outcomes: tuple[int, ...]  # ScenarioOutcome values, one per (champion, pool_member) run
    outcome_stdev: float
    any_decisive_event: bool  # True if any run had a non-NEUTRAL outcome or a foul


# A run is "decisive" if the outcome differs from NEUTRAL or a foul fired —
# matches item 14's "shot, turnover, goal, or foul in any run" DEAD test.
_NEUTRAL = int(ScenarioOutcome.NEUTRAL)

# Outcome stdev at or below this (in ordinal units) across all runs counts
# as DETERMINED — every run landed on essentially the same result.
_DETERMINED_STDEV = 0.01

# Outcome stdev at or above this counts as NOISY — the family's noise floor
# decision is deferred to the bank curator (see module docstring); this
# threshold only decides whether NOISY is reported at all.
_NOISY_STDEV = 1.5


def screen_scenario(
    bench_scenario: BenchScenario,
    *,
    champion_config: str,
    pool_configs: Sequence[str] = DEFAULT_SCREEN_POOL,
    horizon_s: float = 20.0,
) -> DynamicScreenResult:
    """Play `bench_scenario` forward with the champion against itself and
    against each of `pool_configs`, classify the outcome spread.

    Champion-vs-self is always included first (needed to measure the
    family's own noise floor per item 14) — `pool_configs` defaults to a
    single additional pairing so this stays cheap; a bank-curation pass
    building the full v1 bank would pass a larger pool.
    """
    opponents = (champion_config, *pool_configs)
    outcomes: list[int] = []
    any_decisive = False

    for opponent_config in opponents:
        result = score_scenario(
            bench_scenario,
            candidate_config=champion_config,
            opponent_config=opponent_config,
            horizon_s=horizon_s,
        )
        outcomes.append(int(result.outcome))
        if result.outcome != ScenarioOutcome.NEUTRAL or result.foul:
            any_decisive = True

    stdev = statistics.pstdev(outcomes) if len(outcomes) > 1 else 0.0

    if not any_decisive:
        verdict = ScreenVerdict.DEAD
    elif stdev <= _DETERMINED_STDEV:
        verdict = ScreenVerdict.DETERMINED
    elif stdev >= _NOISY_STDEV:
        verdict = ScreenVerdict.NOISY
    else:
        verdict = ScreenVerdict.INFORMATIVE

    return DynamicScreenResult(
        scenario_id=bench_scenario.scenario_id,
        verdict=verdict,
        outcomes=tuple(outcomes),
        outcome_stdev=stdev,
        any_decisive_event=any_decisive,
    )
