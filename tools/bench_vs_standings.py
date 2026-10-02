#!/usr/bin/env python3
# ruff: noqa: E402
"""Does the scenario bench rank strategies the way round-robins do?

Scores every strategy in a round-robin's `summary.json` on the same random sample
of a scenario bank, against fixed opponents, and compares the strategies' mean
bench outcome with their standings (points per match, 3 a win and 1 a draw, and
goal difference per match) by Spearman rank correlation. With `--also-summary`,
the same correlation between two round-robins' standings is printed too: the bench
can't be expected to agree with the standings better than they agree with
themselves.

    pixi run python tools/bench_vs_standings.py \\
        --summary replays/tournament_<id>/summary.json \\
        --also-summary replays/tournament_<older id>/summary.json \\
        --bank utama_core/scenario_bench/banks/bank_v5.json --sample 200 \\
        --opponents counter_press high_line_zone --workers 15
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.scenario_bench.bench_scenario import BenchScenario, load_bank
from utama_core.scenario_bench.scenario_scorer import score_scenario


def standings(summary: dict) -> dict[str, dict[str, float]]:
    """Points and goal difference per match, per strategy, from a round-robin summary."""
    out = {}
    for name, s in summary["strategies"].items():
        n = s["matches"] or 1
        out[name] = {
            "points": (3 * s["wins"] + s["draws"]) / n,
            "goal_diff": (s["goals_for"] - s["goals_against"]) / n,
        }
    return out


def _ranks(values: list[float]) -> list[float]:
    """1-based ranks, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var if var else 0.0


def _play(job: tuple[str, str, BenchScenario]) -> tuple[str, int, bool]:
    strategy, opponent, scenario = job
    result = score_scenario(scenario, candidate_config=strategy, opponent_config=opponent)
    return strategy, int(result.outcome), result.error is None


def bench_scores(
    strategies: list[str], opponents: list[str], scenarios: list[BenchScenario], workers: int, play=_play
) -> dict[str, float]:
    """Mean outcome per strategy over every (opponent, scenario), errored runs left out."""
    jobs = [(s, o, sc) for s in strategies for o in opponents for sc in scenarios]
    outcomes: dict[str, list[int]] = {s: [] for s in strategies}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(play, jobs, chunksize=4))
    else:
        results = [play(j) for j in jobs]
    for strategy, outcome, ok in results:
        if ok:
            outcomes[strategy].append(outcome)
    return {s: statistics.fmean(v) if v else 0.0 for s, v in outcomes.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True, help="a round-robin's summary.json")
    parser.add_argument("--also-summary", type=Path, default=None, help="another round-robin, for the ceiling")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200, help="bank starts per strategy and opponent (default 200)")
    parser.add_argument("--opponents", nargs="+", required=True, help="fixed opponents every strategy plays")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="write the per-strategy table as JSON here")
    args = parser.parse_args()

    table = standings(json.loads(args.summary.read_text()))
    strategies = sorted(table)
    _bank_id, bank = load_bank(args.bank)
    sample = random.Random(0).sample(bank, min(args.sample, len(bank)))
    print(f"{len(strategies)} strategies x {len(args.opponents)} opponents x {len(sample)} starts", flush=True)

    scores = bench_scores(strategies, args.opponents, sample, args.workers)
    rows = sorted(strategies, key=lambda s: -scores[s])
    print(f"\n{'strategy':34s} {'bench':>7s} {'points':>7s} {'goal diff':>9s}")
    for s in rows:
        print(f"{s:34s} {scores[s]:+7.3f} {table[s]['points']:7.2f} {table[s]['goal_diff']:+9.2f}")

    bench = [scores[s] for s in strategies]
    for key in ("points", "goal_diff"):
        print(f"Spearman bench vs {key}: {spearman(bench, [table[s][key] for s in strategies]):+.2f}")
    if args.also_summary:
        other = standings(json.loads(args.also_summary.read_text()))
        common = [s for s in strategies if s in other]
        for key in ("points", "goal_diff"):
            rho = spearman([table[s][key] for s in common], [other[s][key] for s in common])
            print(f"Spearman round-robin vs round-robin, {key}: {rho:+.2f} (the ceiling)")
    if args.output:
        args.output.write_text(json.dumps({"bench": scores, "standings": table}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
