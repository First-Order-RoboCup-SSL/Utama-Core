"""How often evaluate.py's rule gets it right: the rule played on simulated match results.

    pixi run python tools/evaluation/verdict_sim.py [--trials 4000] [--opponents 4 5] [--sd 1.6]

No matches are played. A candidate and the reference meet each opponent once per round (same seed and
colour, so their results are correlated `rho`: ~0.9 for a small edit of one strategy, ~0 for
unrelated strategies); every match's goal difference has SD `sd`. The rule is `ratings.decide`, the
one evaluate.py runs: a screen of ROUNDS rounds that only drops, then ROUNDS fresh rounds that accept
at p <= ALPHA. Prints, per true effect (goals per match) and correlation, the share of candidates
called better, dropped and undecided, and the mean number of rounds played.

What it leaves out: a candidate's effect is the same against every opponent (no matchups), and diffs
are normal (real ones are whole goals, maybe heavy-tailed). The test (`tests/tools/test_verdict_sim.py`)
pins the claim the docs make: an equal strategy is called better at most ALPHA of the time.
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


def simulate(
    effect: float, rho: float, n_opponents: int, trials: int, seed: int = 0, sd: float = ratings.NOISE_SD
) -> dict:
    """Share of `trials` candidates called better / dropped / undecided, and their mean rounds."""
    rng = np.random.default_rng(seed)
    sd_diff = sd * np.sqrt(2 * (1 - rho))  # SD of one diff: candidate's margin minus the reference's
    out = {"better": 0, "dropped": 0, "undecided": 0, "rounds": 0}
    for _ in range(trials):
        verdict, k = None, 0
        for confirming in (False, True):
            diffs: list[float] = []  # each stage decides on its own seeds' diffs only
            for j in range(1, ratings.ROUNDS + 1):
                k += 1
                diffs += list(rng.normal(effect, sd_diff, size=n_opponents))
                verdict = ratings.decide(ratings.paired_test(diffs), j, confirming)
                if verdict is not None:
                    break
            if verdict != "confirm":
                break
        out[verdict] += 1
        out["rounds"] += k
    return {key: value / trials for key, value in out.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--opponents", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--sd", type=float, default=ratings.NOISE_SD)
    args = parser.parse_args(argv)
    print(f"ROUNDS {ratings.ROUNDS}, ALPHA {ratings.ALPHA}, match SD {args.sd}")
    print(f"{'opponents':>9} {'rho':>4} {'effect':>6} {'better':>7} {'dropped':>8} {'undecided':>9} {'rounds':>6}")
    for n in args.opponents:
        for rho in (0.0, 0.5, 0.9):
            for effect in (-0.5, 0.0, 0.3, 0.5, 1.0):
                r = simulate(effect, rho, n, args.trials, sd=args.sd)
                print(
                    f"{n:>9} {rho:>4.1f} {effect:>+6.1f} {r['better']:>7.1%} {r['dropped']:>8.1%} "
                    f"{r['undecided']:>9.1%} {r['rounds']:>6.1f}"
                )


if __name__ == "__main__":
    main()
