"""elo.py — Compute Elo ratings from one or more tournament.py summary.json files.

Run:
    pixi run python elo.py replays/tournament_<id>/summary.json [more summaries...]

What this does
---------------
Reads each summary.json's "results" list (config_a/config_b/score_a/score_b
per match, in the order tournament.py recorded them) and replays it through a
standard Elo update: every match is one game between two configs, win/draw/
loss decided by final score, K-factor fixed (see K below). Multiple summary
files are concatenated in the order given, so e.g. a --both-sides run and a
later re-run after a fix both feed the same continuously-updated ratings.

Deliberately the smallest mechanism that answers "who's ahead, and by how
much, accounting for who they played" — a straight port of the textbook Elo
update (no Glicko/TrueSkill rating-uncertainty machinery, no separate
per-surface ratings) over tournament.py's existing summary.json, not a new
persistence layer. Prints a final ratings table and writes a per-match
rating-history JSON (elo_history.json, one row per match with each config's
rating *after* that match) for plot_elo.py to consume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STARTING_RATING = 1000.0
K = 24.0  # standard-ish K-factor; no rating-uncertainty tapering (see module docstring)


def _short_name(config_name: str) -> str:
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(score_a: int, score_b: int) -> float:
    if score_a > score_b:
        return 1.0
    if score_b > score_a:
        return 0.0
    return 0.5


def compute_elo(all_results: list[dict]) -> tuple[dict[str, float], list[dict]]:
    """Returns (final_ratings, history) where history is one row per match:
    {"match_index", "config_a", "config_b", "score_a", "score_b",
     "rating_a_after", "rating_b_after"}.
    """
    ratings: dict[str, float] = {}
    history: list[dict] = []

    for i, r in enumerate(all_results):
        a, b = _short_name(r["config_a"]), _short_name(r["config_b"])
        ratings.setdefault(a, STARTING_RATING)
        ratings.setdefault(b, STARTING_RATING)

        exp_a = expected_score(ratings[a], ratings[b])
        act_a = actual_score(r["score_a"], r["score_b"])
        delta = K * (act_a - exp_a)
        ratings[a] += delta
        ratings[b] -= delta

        history.append(
            {
                "match_index": i,
                "config_a": a,
                "config_b": b,
                "score_a": r["score_a"],
                "score_b": r["score_b"],
                "rating_a_after": ratings[a],
                "rating_b_after": ratings[b],
            }
        )

    return ratings, history


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("Usage: pixi run python elo.py <summary.json> [more summary.json ...]")

    all_results: list[dict] = []
    for path in paths:
        with open(path) as f:
            summary = json.load(f)
        all_results.extend(summary["results"])

    ratings, history = compute_elo(all_results)

    print(f"Elo after {len(all_results)} matches from {len(paths)} summary file(s):\n")
    for name in sorted(ratings, key=lambda n: -ratings[n]):
        print(f"  {name:<20} {ratings[name]:7.1f}")

    out_path = paths[0].parent / "elo_history.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "starting_rating": STARTING_RATING,
                "k_factor": K,
                "source_files": [str(p) for p in paths],
                "final_ratings": ratings,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"\nRating history: {out_path}")


if __name__ == "__main__":
    main()
