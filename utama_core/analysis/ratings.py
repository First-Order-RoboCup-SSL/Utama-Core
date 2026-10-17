"""Strategy ratings with uncertainty, from match margins.

Model: a match's margin (goal difference, config_a minus config_b) is strength(a) - strength(b)
plus noise, normal with SD `noise_sd`; every strength has a normal prior, mean 0 and SD
`PRIOR_SD`. The posterior is normal and closed-form (Bayesian ridge regression), so a strength
comes with its SD, and two strengths with the probability one is the larger.

A strength is in goals per match against an opponent of strength 0. The prior centres the
ratings of whoever has played on 0, so compare strengths within one fit (`p_better`), not
across fits.

`noise_sd` is fitted from the residuals, shrunk toward `NOISE_SD` (the per-match spread of
goal difference left after strengths in `tournament_20261016_201213`, 66 matches) with the
weight of `NOISE_PRIOR_N` matches, so a few matches cannot make it tiny.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PRIOR_SD = 1.0  # goals per match; strengths in the 2026-10-16 baseline spread with SD ~0.7
NOISE_SD = 1.6
NOISE_PRIOR_N = 10


@dataclass
class Ratings:
    names: list[str]
    mean: np.ndarray
    cov: np.ndarray
    noise_sd: float
    n_matches: int

    def _i(self, name: str) -> int:
        return self.names.index(name)

    def strength(self, name: str) -> tuple[float, float]:
        """(mean, SD) of `name`'s strength."""
        i = self._i(name)
        return float(self.mean[i]), math.sqrt(self.cov[i, i])

    def diff(self, x: str, y: str) -> tuple[float, float]:
        """(mean, SD) of strength(x) - strength(y): x's expected margin over y."""
        i, j = self._i(x), self._i(y)
        var = self.cov[i, i] + self.cov[j, j] - 2 * self.cov[i, j]
        return float(self.mean[i] - self.mean[j]), math.sqrt(max(var, 0.0))

    def p_better(self, x: str, y: str) -> float:
        """Posterior probability that x is stronger than y."""
        m, s = self.diff(x, y)
        return 0.5 * (1 + math.erf(m / (s * math.sqrt(2)))) if s > 0 else float(m > 0)


def fit(
    matches: list[tuple[str, str, float]],
    names: list[str] | None = None,
    prior_sd: float = PRIOR_SD,
    noise_sd: float | None = None,
) -> Ratings:
    """Ratings from `(config_a, config_b, margin)` triples. `names` adds strategies with no
    matches yet (they keep the prior). `noise_sd`, if given, is used as is instead of fitted."""
    names = sorted(set(names or []) | {a for a, _, _ in matches} | {b for _, b, _ in matches})
    ix = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(matches), len(names)))
    y = np.array([m for _, _, m in matches], float)
    for row, (a, b, _) in enumerate(matches):
        A[row, ix[a]] += 1.0
        A[row, ix[b]] -= 1.0
    prior_precision = np.eye(len(names)) / prior_sd**2

    def posterior(sd: float) -> tuple[np.ndarray, np.ndarray]:
        cov = np.linalg.inv(A.T @ A / sd**2 + prior_precision)
        return cov @ A.T @ y / sd**2, cov

    sd = noise_sd
    if sd is None:
        sd = NOISE_SD
        for _ in range(5):  # a few fixed-point steps: the residuals depend on the fit
            mean, _ = posterior(sd)
            rss = float(np.sum((y - A @ mean) ** 2))
            dof = max(len(matches) - len(names) + 1, 0)
            sd = math.sqrt((NOISE_PRIOR_N * NOISE_SD**2 + rss) / (NOISE_PRIOR_N + dof))
    mean, cov = posterior(sd)
    return Ratings(names, mean, cov, sd, len(matches))


PAIRED_PRIOR_N = 6  # weight, in paired differences, of the prior that pairing removes no noise


def paired_difference(diffs: list[float], head_to_head: list[float], noise_sd: float) -> tuple[float, float]:
    """(mean, SD) of strength(candidate) - strength(baseline) from paired matches.

    `diffs`: for each opponent, seed and colour both played, the candidate's goal difference minus
    the baseline's. Same opponent and same noise, so much of the noise cancels: all of it for
    identical code, little for unrelated strategies. Their variance is estimated from the diffs
    themselves, shrunk toward 2 * noise_sd^2 (no cancelling) with the weight of
    `PAIRED_PRIOR_N` diffs, so a few equal results cannot make it tiny.
    `head_to_head`: the candidate's margins against the baseline, each with variance noise_sd^2.
    The prior on the difference is normal, mean 0, SD sqrt(2) * PRIOR_SD (two strengths)."""
    precision = 1 / (2 * PRIOR_SD**2)
    weighted = 0.0
    if diffs:
        d = np.asarray(diffs, float)
        ss = float(np.sum((d - d.mean()) ** 2))
        var_d = (PAIRED_PRIOR_N * 2 * noise_sd**2 + ss) / (PAIRED_PRIOR_N + len(d) - 1)
        precision += len(d) / var_d
        weighted += d.sum() / var_d
    if head_to_head:
        precision += len(head_to_head) / noise_sd**2
        weighted += float(np.sum(head_to_head)) / noise_sd**2
    return weighted / precision, math.sqrt(1 / precision)


def p_positive(mean: float, sd: float) -> float:
    return 0.5 * (1 + math.erf(mean / (sd * math.sqrt(2))))
