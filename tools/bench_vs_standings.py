#!/usr/bin/env python3
# ruff: noqa: E402
"""Does the scenario bench predict round-robin goal difference?

Plays a baseline and every candidate once on each start of a bank, each start against the
opponent it came from (`scenario_bench.start_opponent`), and compares per candidate the
bench's effect on goal difference per match (`scenario_bench.situation_effect`, candidate
minus baseline) with the round-robin's: the candidate's mean goal difference per match minus
the baseline's, over the opponents the bench played them against. A start whose opponent is
the candidate or the baseline is left out of that candidate's effect, so both numbers are
against the same opponents. Prints Pearson and Spearman correlations over the candidates.

The round-robin's own situation rates (sim) are the ones that should predict it; others (real
Division B rates) are printed beside them, compared with the same round-robin for reference.

    pixi run python tools/bench_vs_standings.py \\
        --summary replays/tournament_20261011_230119/summary.json \\
        --bank utama_core/scenario_bench/banks/bank_v8.json --baseline split_shape \\
        --candidates overload_press press_and_pass counter_flow clear_press_plus high_press decoy_and_overload \\
        --weights utama_core/scenario_bench/banks/situation_rates_sim.json \\
                  utama_core/scenario_bench/banks/situation_rates_real.json \\
        --workers 15 --output bench_vs_standings.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.scenario_bench import (
    _start_result,
    load_weights,
    situation_effect,
    start_opponent,
)
from utama_core.analysis.goal_ledger import WINDOW_S
from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    enable_sim_reuse,
)
from utama_core.scenario_bench.scenario_scorer import _resolve_config_name
from utama_core.scenario_bench.start import BenchScenario, load_bank


def goal_difference(summary: dict, config: str, opponents: set[str]) -> float:
    """`config`'s mean goal difference per match against `opponents`, from a round-robin's
    `summary.json` (each opponent weighted equally)."""
    per_opponent: dict[str, list[int]] = {}
    for r in summary["results"]:
        if r["config_a"] == config and r["config_b"] in opponents:
            per_opponent.setdefault(r["config_b"], []).append(r["score_a"] - r["score_b"])
        elif r["config_b"] == config and r["config_a"] in opponents:
            per_opponent.setdefault(r["config_a"], []).append(r["score_b"] - r["score_a"])
    return statistics.fmean(statistics.fmean(v) for v in per_opponent.values())


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


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    var = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return cov / var if var else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def _play(job: tuple[str, BenchScenario, float]) -> tuple[str, str, dict]:
    config, bench_scenario, horizon_s = job
    enable_sim_reuse()  # a worker plays many starts: keep its sim subprocess between them
    return (
        config,
        bench_scenario.scenario_id,
        _start_result(bench_scenario, config, start_opponent(bench_scenario), horizon_s),
    )


def paired_rows(starts: list[BenchScenario], results: dict, candidate: str, baseline: str) -> list[dict]:
    """`situation_effect`'s rows for `candidate` against `baseline`, from `results[config][scenario
    id]`, leaving out starts played against either of them."""
    rows = []
    for bs in starts:
        if start_opponent(bs) in (candidate, baseline):
            continue
        cand, base = results[candidate][bs.scenario_id], results[baseline][bs.scenario_id]
        rows.append(
            {
                "situation": bs.provenance.situation,
                "candidate_outcomes": [cand["outcome"]],
                "baseline_outcomes": [base["outcome"]],
                "candidate_errors": [cand["error"]] if cand["error"] else [],
                "baseline_errors": [base["error"]] if base["error"] else [],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True, help="the round-robin's summary.json")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True, help="situation rates files, each reported")
    parser.add_argument("--horizon", type=float, default=WINDOW_S)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="write per-start outcomes and the table here")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    baseline = _resolve_config_name(args.baseline)
    candidates = [_resolve_config_name(c) for c in args.candidates]
    weights = [load_weights(w) for w in args.weights]
    _bank_id, bank = load_bank(args.bank)
    starts = [bs for bs in bank if bs.provenance.situation and start_opponent(bs) is not None]
    print(f"{len(starts)} of {len(bank)} starts playable here; {len(candidates) + 1} configs", flush=True)

    jobs = [(config, bs, args.horizon) for config in [baseline, *candidates] for bs in starts]
    results: dict[str, dict[str, dict]] = {config: {} for config in [baseline, *candidates]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (config, sid, result) in enumerate(pool.map(_play, jobs, chunksize=4), 1):
            results[config][sid] = result
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} runs", flush=True)

    table = {}
    for candidate in candidates:
        rows = paired_rows(starts, results, candidate, baseline)
        opponents = {start_opponent(bs) for bs in starts} - {candidate, baseline}
        table[candidate] = {
            "n_starts": len(rows),
            "actual": goal_difference(summary, candidate, opponents) - goal_difference(summary, baseline, opponents),
            "effects": [situation_effect(rows, w) for w in weights],
        }

    names = [Path(w["path"]).stem.removeprefix("situation_rates_") for w in weights]
    print(f"\nGoal difference per match relative to {baseline}, against the same opponents")
    print(f"{'candidate':42s} {'n':>4s} {'round-robin':>11s} " + " ".join(f"{'bench, ' + n:>16s}" for n in names))
    for candidate, t in table.items():
        cells = []
        for effect in t["effects"]:
            total = effect["total"]
            se = total["net_per_match_se"]
            cells.append(
                f"{total['net_per_match']:+7.2f} ± {se:4.2f}" if se is not None else f"{total['net_per_match']:+7.2f}"
            )
        print(f"{candidate:42s} {t['n_starts']:4d} {t['actual']:+11.2f} " + " ".join(f"{c:>16s}" for c in cells))
    actual = [t["actual"] for t in table.values()]
    print()
    for i, name in enumerate(names):
        predicted = [t["effects"][i]["total"]["net_per_match"] for t in table.values()]
        print(
            f"bench ({name} weights) vs round-robin over {len(candidates)} candidates: "
            f"Pearson {pearson(predicted, actual):+.2f}, Spearman {spearman(predicted, actual):+.2f}"
        )
    if args.output:
        payload = {"baseline": baseline, "weights": weights, "table": table, "results": results}
        args.output.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
