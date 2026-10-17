"""Expected standings: each strategy's strength in a round-robin, from goals and from shots on target.

A strength is a least-squares fit of per-match differences (ours minus theirs) as strength(a) -
strength(b), summing to zero over the strategies: the margin a strategy wins by against an average
opponent of the run, per match. `goals` fits goal difference; `shots` fits shot-on-target
difference (`MatchStats.shots`: hard balls from the attacking third heading into the goal mouth).

Shots on target are an outcome estimator, not a signal (`docs/signals.md`, "Ranking"): on the
66-match noisy baseline of 2026-10-16, strengths fitted on half the matches predicted the other
half's goal differences better from shots (r 0.28) than from goals (r 0.17), and agreed between
halves more (0.53 against 0.28). On the deterministic sim of 2026-10-14 the two were level. Rank by
both; decide with goals over enough matches.
"""

from __future__ import annotations

import numpy as np

ESTIMATORS = ("goals", "shots")


def _difference(r: dict, estimator: str) -> float:
    if estimator == "goals":
        return r["score_a"] - r["score_b"]
    shots = (r.get("stats") or {}).get("shots") or {}
    return shots.get("friendly", 0) - shots.get("enemy", 0)


def strengths(results: list[dict], estimator: str) -> dict[str, float]:
    """Each config's strength over `summary.json`-shaped `results`, by `estimator`."""
    names = sorted({r["config_a"] for r in results} | {r["config_b"] for r in results})
    if not results:
        return {}
    ix = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(results), len(names)))
    y = np.array([_difference(r, estimator) for r in results], float)
    for row, r in enumerate(results):
        A[row, ix[r["config_a"]]] += 1.0
        A[row, ix[r["config_b"]]] -= 1.0
    # Minimum-norm solution: orthogonal to the all-ones null space, so strengths sum to zero.
    s = np.linalg.lstsq(A, y, rcond=None)[0]
    return {n: round(float(s[ix[n]]), 2) for n in names}


def expected_standings(results: list[dict]) -> dict[str, dict[str, float]]:
    """{config: {"goals": strength, "shots": strength}}."""
    by = {e: strengths(results, e) for e in ESTIMATORS}
    return {n: {e: by[e][n] for e in ESTIMATORS} for n in by["goals"]}
