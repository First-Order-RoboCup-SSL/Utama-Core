"""How often evaluate.py's stopping rule gets it right: the rule played on simulated match results.

    pixi run python tools/evaluation/verdict_sim.py [--trials 4000] [--opponents 4 5]

No matches are played. A candidate and the reference meet each opponent once per round (same seed and
colour, so their results are correlated `rho`: ~0.9 for a small edit of one strategy, ~0 for
unrelated strategies) and each other once; every match's goal difference has SD `ratings.NOISE_SD`.
The rule is evaluate.py's: `ratings.paired_difference`, better at P >= PROMOTE, dropped at P <= DROP,
undecided after MAX_ROUNDS. Prints, per true effect (goals per match) and correlation, the share of
candidates called better and dropped, and the mean number of rounds.

What it leaves out: the noise SD is given, not fitted (evaluate.py fits it from the store), and a
candidate's effect is the same against every opponent (no matchups). The test
(`tests/tools/test_verdict_sim.py`) pins the claim the docs make: two equal strategies are called
better rarely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis import ratings  # noqa: E402

# evaluate.py's rule; kept here so this needs no simulator (tests check they agree)
PROMOTE = 0.99
DROP = 0.2
MAX_ROUNDS = 6


def simulate(
    effect: float, rho: float, n_opponents: int, trials: int, seed: int = 0, sd: float = ratings.NOISE_SD
) -> dict:
    """Share of `trials` candidates called better / dropped / undecided, and their mean rounds."""
    rng = np.random.default_rng(seed)
    cov = sd**2 * np.array([[1.0, rho], [rho, 1.0]])
    out = {"better": 0, "dropped": 0, "undecided": 0, "rounds": 0}
    for _ in range(trials):
        diffs: list[float] = []
        h2h: list[float] = []
        for k in range(1, MAX_ROUNDS + 1):
            pair = rng.multivariate_normal([effect, 0.0], cov, size=n_opponents)
            diffs += list(pair[:, 0] - pair[:, 1])
            h2h.append(effect + rng.normal(0.0, sd))
            margin, msd = ratings.paired_difference(diffs, h2h, sd)
            p = ratings.p_positive(margin, msd)
            verdict = "better" if p >= PROMOTE else "dropped" if p <= DROP else None
            if verdict is None and k == MAX_ROUNDS:
                verdict = "undecided"
            if verdict is not None:
                out[verdict] += 1
                out["rounds"] += k
                break
    return {key: value / trials for key, value in out.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--opponents", type=int, nargs="+", default=[4, 5])
    args = parser.parse_args(argv)
    print(f"{'opponents':>9} {'rho':>4} {'effect':>6} {'better':>7} {'dropped':>8} {'undecided':>9} {'rounds':>6}")
    for n in args.opponents:
        for rho in (0.0, 0.5, 0.9):
            for effect in (-1.0, -0.5, 0.0, 0.5, 1.0):
                r = simulate(effect, rho, n, args.trials)
                print(
                    f"{n:>9} {rho:>4.1f} {effect:>+6.1f} {r['better']:>7.1%} {r['dropped']:>8.1%} "
                    f"{r['undecided']:>9.1%} {r['rounds']:>6.1f}"
                )


if __name__ == "__main__":
    main()
