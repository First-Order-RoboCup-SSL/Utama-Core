"""The verdict's test: is the candidate's mean paired difference above zero?

A paired difference ("diff") is the candidate's goal difference minus the reference's against the
same opponent, with the same seed and colour (`tools/evaluation/evaluate.py`'s `paired`). The same
opponent cancels its strength, and the same noise cancels part of the luck: much of it for a small
edit, little for an unrelated strategy. The test is a one-sided paired t-test on the diffs: it
estimates their spread from the diffs themselves, so it needs no assumed noise level, and a small
sample is paid for by the t distribution's wider tails, not by a hand-set prior.

The rule built on it (`decide`): a screen on seeds 1..ROUNDS that only drops (mean diff <= 0), then
a confirm stage on fresh seeds that accepts at p <= ALPHA. `tools/evaluation/verdict_sim.py` plays it
on simulated results.

Until 2026-11-12 this module held a Bayesian rating model with a prior on the diffs' spread
(weight 6) and head-to-head matches in the verdict; it was replaced by this test (git history).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from scipy import stats

# Rounds per stage (screen, then confirm). Provisional: set it from the measured spread s of real
# diffs so a 0.3-goal improvement is found 80% of the time (rough: s 0.5 -> 3, 0.7 -> 7, 1.0 -> 14).
ROUNDS = 6
# One-sided error bar of the confirm stage: the share of equal candidates called better. Provisional:
# set it from how often edits turn out better (the rarer, the smaller ALPHA must be).
ALPHA = 0.05
NOISE_SD = 1.6  # per-match spread of goal difference (tournament_20261016_201213); verdict_sim's default


@dataclass(frozen=True)
class PairedTest:
    n: int  # number of diffs
    mean: float  # mean diff, goals per match
    se: float  # its standard error
    p: float  # one-sided p-value for "mean > 0": how often an equal candidate would look this good


def paired_test(diffs: list[float]) -> PairedTest:
    """The one-sided paired t-test of `diffs`. Fewer than two diffs, or none above zero, give p = 1."""
    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0
    if n < 2:
        return PairedTest(n, mean, math.inf, 1.0)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0:  # every diff the same: identical code (all zero), or an exact, repeated margin
        return PairedTest(n, mean, 0.0, 0.0 if mean > 0 else 1.0)
    return PairedTest(n, mean, se, float(stats.t.sf(mean / se, n - 1)))


def decide(test: PairedTest, rounds_in_stage: int, confirming: bool) -> Optional[str]:
    """evaluate.py's rule after a round of a stage: "dropped" (mean diff <= 0, from the stage's second
    round), "confirm" (the screen survived), "better" (confirm stage, p <= ALPHA), "undecided"
    (confirm stage, p > ALPHA), or None to play another round. The screen never accepts."""
    if rounds_in_stage >= 2 and test.n and test.mean <= 0:
        return "dropped"
    if rounds_in_stage < ROUNDS:
        return None
    if not confirming:
        return "confirm"
    return "better" if test.p <= ALPHA else "undecided"
