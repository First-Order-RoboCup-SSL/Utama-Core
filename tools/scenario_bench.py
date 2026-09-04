#!/usr/bin/env python3
# ruff: noqa: E402
"""Outer-loop scenario bench: score a candidate strategy against a baseline
on a bank of short scenarios, always as a paired differential.

This is the "outer loop, fast half" from `docs/roadmap.md` item 14 — a
benchmark like `tools/motion_planning_benchmark.py`, non-blocking, reporting
numbers rather than gating a merge. See that roadmap item's 2026-09-04
design pass for the full rationale behind every design choice below;
this docstring only summarizes what the tool actually does.

Bank sources, in the order this tool can use them today:
  - Hand-authored anchors (`utama_core.replay.hand_authored_scenarios`) —
    always available, no dependency on any replay data.
  - Harvested restart-transition scenarios (`--harvest-from RUN_DIR`) — only
    from a tagged, trustworthy tournament run (`.stats.json` with zero
    `stall_events`, see `utama_core.replay.scenario_harvester`'s module
    docstring for why this gate exists). NOT run by this tool — point it at
    an already-completed `tournament.py` run directory.

For each scenario, both `--candidate` and `--baseline` play the SAME
scenario against the SAME `--opponent` (paired comparison, per item 14's
"Compute discipline" section — the variance of the *difference* is what
matters, not either absolute score). The report is the per-scenario and
per-family paired outcome delta, never an absolute score.

`--save-bank PATH` persists the currently-loaded scenario set (hand-authored
+ optional `--harvest-from`, after `--families` filtering) to a single JSON
file via `bench_scenario.save_bank` — a few KB even for hundreds of
scenarios, since a scenario is field state only, no trajectories. `--load-bank
PATH` loads scenarios from a previously-saved bank instead of hand-authored/
`--harvest-from` (the two are mutually exclusive as *sources*: a loaded bank
is meant to be the frozen, already-screened set a prior save produced, not a
starting point to silently merge fresh sources into — see item 14's
"Immutable per bank version" note). Still not built: automatic lifecycle
promotion (candidate -> validated -> active) or a ladder (slow half) — this
tool can freeze *what* the bank contains, not yet *which scenarios in it are
trustworthy enough to score*; that curation is still a manual step (e.g. run
`--dynamic-screen`, decide by hand, then `--save-bank` only the survivors).

Run from the repository root, for example:

    pixi run python tools/scenario_bench.py \\
        --candidate build_tiki_taka_kernel_strategy \\
        --baseline build_default_kernel_strategy \\
        --opponent build_low_block_kernel_strategy

    pixi run python tools/scenario_bench.py --list-scenarios

    pixi run python tools/scenario_bench.py \\
        --candidate build_tiki_taka_kernel_strategy --baseline build_default_kernel_strategy \\
        --opponent build_low_block_kernel_strategy --harvest-from replays/tournament_20260905_090000

    # Freeze a bank from a harvest for reuse across sessions:
    pixi run python tools/scenario_bench.py --harvest-from replays/tournament_20260905_090000 \\
        --save-bank utama_core/replay/banks/bank_v1.json --bank-id v1 --list-scenarios

    # Score against the frozen bank later, without re-harvesting:
    pixi run python tools/scenario_bench.py --load-bank utama_core/replay/banks/bank_v1.json \\
        --candidate build_tiki_taka_kernel_strategy --baseline build_default_kernel_strategy \\
        --opponent build_low_block_kernel_strategy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.replay.bench_scenario import BenchScenario, load_bank, save_bank
from utama_core.replay.dynamic_screen import (
    DEFAULT_SCREEN_POOL,
    ScreenVerdict,
    screen_scenario,
)
from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios
from utama_core.replay.scenario_harvester import harvest_run_dir
from utama_core.replay.scenario_scorer import ScenarioOutcome, score_scenario

SCHEMA_VERSION = 1


def _git_revision() -> str | None:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_bank(args: argparse.Namespace) -> tuple[list[BenchScenario], dict]:
    harvest_report: dict = {}

    if args.load_bank is not None:
        # A persisted bank replaces the hand-authored + harvest sources
        # entirely rather than merging with them — it's meant to be the
        # frozen, already-screened set a prior `--save-bank` produced (see
        # `bench_scenario.save_bank`'s "immutable per bank version" note),
        # not a starting point to silently mix fresh sources into.
        _bank_id, scenarios = load_bank(args.load_bank)
    else:
        scenarios = list(all_hand_authored_scenarios())
        if args.harvest_from is not None:
            harvested, harvest_report = harvest_run_dir(
                args.harvest_from,
                evaluator_version=_git_revision() or "unknown",
            )
            scenarios.extend(harvested)

    if args.families:
        wanted = set(args.families)
        scenarios = [s for s in scenarios if s.provenance.family.value in wanted]

    if args.save_bank is not None:
        save_bank(scenarios, args.save_bank, bank_id=args.bank_id or args.save_bank.stem)
        print(f"Saved {len(scenarios)} scenarios to {args.save_bank} (bank_id={args.bank_id or args.save_bank.stem})")

    return scenarios, harvest_report


def _run_dynamic_screen(scenarios: list[BenchScenario], *, horizon_s: float) -> list[dict]:
    results = []
    for bench_scenario in scenarios:
        result = screen_scenario(
            bench_scenario,
            champion_config="build_default_kernel_strategy",
            pool_configs=DEFAULT_SCREEN_POOL,
            horizon_s=horizon_s,
        )
        results.append(
            {
                "scenario_id": result.scenario_id,
                "verdict": result.verdict.value,
                "outcomes": list(result.outcomes),
                "outcome_stdev": result.outcome_stdev,
            }
        )
    return results


def _score_paired(
    scenarios: list[BenchScenario],
    *,
    candidate: str,
    baseline: str,
    opponent: str,
    horizon_s: float,
) -> list[dict]:
    rows = []
    total = len(scenarios)
    for index, bench_scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{total}] {bench_scenario.scenario_id} ({bench_scenario.provenance.family.value})", flush=True)

        candidate_result = score_scenario(
            bench_scenario, candidate_config=candidate, opponent_config=opponent, horizon_s=horizon_s
        )
        baseline_result = score_scenario(
            bench_scenario, candidate_config=baseline, opponent_config=opponent, horizon_s=horizon_s
        )

        delta = int(candidate_result.outcome) - int(baseline_result.outcome)
        print(
            f"  candidate={candidate_result.outcome.name} baseline={baseline_result.outcome.name} delta={delta:+d}",
            flush=True,
        )

        rows.append(
            {
                "scenario_id": bench_scenario.scenario_id,
                "family": bench_scenario.provenance.family.value,
                "trigger": bench_scenario.provenance.trigger.value,
                "perspective": bench_scenario.provenance.perspective,
                "candidate_outcome": candidate_result.outcome.name,
                "baseline_outcome": baseline_result.outcome.name,
                "delta": delta,
                "candidate_foul": candidate_result.foul,
                "baseline_foul": baseline_result.foul,
                "candidate_error": candidate_result.error,
                "baseline_error": baseline_result.error,
            }
        )
    return rows


def _aggregate_by_family(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)

    aggregates = []
    for family, family_rows in grouped.items():
        deltas = [r["delta"] for r in family_rows]
        n = len(deltas)
        mean_delta = sum(deltas) / n if n else 0.0
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        ties = n - wins - losses
        aggregates.append(
            {
                "family": family,
                "n_scenarios": n,
                "mean_delta": mean_delta,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    return sorted(aggregates, key=lambda a: a["family"])


def _markdown_report(payload: dict) -> str:
    lines = [
        "# Scenario bench report",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Git revision: `{payload['git_revision'] or 'unknown'}`  ",
        f"Candidate: `{payload['candidate']}`  ",
        f"Baseline: `{payload['baseline']}`  ",
        f"Opponent: `{payload['opponent']}`  ",
        f"Horizon: {payload['horizon_s']}s  ",
        f"Scenarios scored: {payload['n_scenarios']}",
        "",
        "Delta is `candidate_outcome - baseline_outcome` on the ordinal scale "
        "(GOAL_AGAINST=-3 ... NEUTRAL=0 ... GOAL_FOR=+3), same scenario and opponent for both. "
        "Positive mean delta = candidate did better than baseline on that family. "
        "This is a proxy signal, not an acceptance gate — see roadmap item 14's Goodhart guard.",
        "",
        "## By family",
        "",
        "| Family | N | Mean delta | Wins | Losses | Ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for agg in payload["aggregates"]:
        lines.append(
            f"| {agg['family']} | {agg['n_scenarios']} | {agg['mean_delta']:+.2f} | "
            f"{agg['wins']} | {agg['losses']} | {agg['ties']} |"
        )

    lines.extend(
        ["", "## Per-scenario", "", "| Scenario | Family | Candidate | Baseline | Delta |", "|---|---|---|---|---:|"]
    )
    for row in payload["results"]:
        lines.append(
            f"| {row['scenario_id']} | {row['family']} | {row['candidate_outcome']} | "
            f"{row['baseline_outcome']} | {row['delta']:+d} |"
        )

    errors = [r for r in payload["results"] if r["candidate_error"] or r["baseline_error"]]
    lines.extend(["", "## Errors", ""])
    if not errors:
        lines.append("None.")
    else:
        for row in errors:
            lines.append(
                f"- `{row['scenario_id']}`: candidate={row['candidate_error']} baseline={row['baseline_error']}"
            )

    if payload.get("harvest_report"):
        hr = payload["harvest_report"]
        lines.extend(
            [
                "",
                "## Harvest",
                "",
                f"Matches seen: {hr.get('matches_seen', 0)}, trusted: {hr.get('matches_trusted', 0)}, "
                f"untrusted (dropped): {hr.get('matches_untrusted', 0)}. "
                f"Scenarios harvested: {hr.get('scenarios_harvested', 0)}.",
            ]
        )

    if payload.get("dynamic_screen"):
        lines.extend(["", "## Dynamic screen", "", "| Scenario | Verdict | Outcomes | Stdev |", "|---|---|---|---:|"])
        for row in payload["dynamic_screen"]:
            lines.append(
                f"| {row['scenario_id']} | {row['verdict']} | {row['outcomes']} | {row['outcome_stdev']:.2f} |"
            )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", help="candidate strategy config, e.g. build_tiki_taka_kernel_strategy")
    parser.add_argument("--baseline", help="baseline strategy config to diff against")
    parser.add_argument("--opponent", help="opponent strategy config both candidate and baseline play against")
    parser.add_argument("--horizon", type=float, default=20.0, help="sim seconds ticked per scenario (default 20)")
    parser.add_argument(
        "--harvest-from",
        type=Path,
        default=None,
        help="a completed tournament.py run dir to harvest restart scenarios from",
    )
    parser.add_argument("--families", nargs="+", default=None, help="restrict to these ScenarioFamily values")
    parser.add_argument(
        "--dynamic-screen", action="store_true", help="run the dynamic screen instead of paired scoring"
    )
    parser.add_argument("--list-scenarios", action="store_true", help="print the loaded bank and exit")
    parser.add_argument(
        "--load-bank",
        type=Path,
        default=None,
        help="load scenarios from a persisted bank JSON (see bench_scenario.save_bank) "
        "instead of hand-authored/--harvest-from",
    )
    parser.add_argument(
        "--save-bank",
        type=Path,
        default=None,
        help="write the loaded scenario set (after --families filtering) to this path as a persisted bank JSON",
    )
    parser.add_argument(
        "--bank-id",
        default=None,
        help="bank_id to record when using --save-bank (default: the output filename's stem)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scenario_bench_results"))
    args = parser.parse_args()

    if not args.list_scenarios and not args.dynamic_screen:
        missing = [name for name in ("candidate", "baseline", "opponent") if getattr(args, name) is None]
        if missing:
            parser.error(f"--{', --'.join(missing)} required unless --list-scenarios or --dynamic-screen")

    return args


def main() -> int:
    args = parse_args()
    scenarios, harvest_report = _load_bank(args)

    if args.list_scenarios:
        for bs in scenarios:
            print(
                f"{bs.scenario_id:<45} {bs.provenance.family.value:<28} "
                f"{bs.provenance.trigger.value:<14} lifecycle={bs.lifecycle.value}"
            )
        if harvest_report:
            print(f"\nHarvest: {harvest_report}")
        return 0

    if not scenarios:
        print("No scenarios loaded (check --families / --harvest-from).", file=sys.stderr)
        return 1

    generated_at = datetime.now(timezone.utc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")

    if args.dynamic_screen:
        screen_results = _run_dynamic_screen(scenarios, horizon_s=args.horizon)
        dead = sum(1 for r in screen_results if r["verdict"] == ScreenVerdict.DEAD.value)
        determined = sum(1 for r in screen_results if r["verdict"] == ScreenVerdict.DETERMINED.value)
        noisy = sum(1 for r in screen_results if r["verdict"] == ScreenVerdict.NOISY.value)
        informative = sum(1 for r in screen_results if r["verdict"] == ScreenVerdict.INFORMATIVE.value)
        print(f"\nDynamic screen: {informative} informative, {noisy} noisy, {determined} determined, {dead} dead")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": generated_at.isoformat(),
            "git_revision": _git_revision(),
            "candidate": None,
            "baseline": None,
            "opponent": None,
            "horizon_s": args.horizon,
            "n_scenarios": len(scenarios),
            "aggregates": [],
            "results": [],
            "harvest_report": harvest_report,
            "dynamic_screen": screen_results,
        }
        json_path = args.output_dir / f"scenario_bench_screen_{timestamp}.json"
        json_path.write_text(json.dumps(payload, indent=2))
        md_path = args.output_dir / f"scenario_bench_screen_{timestamp}.md"
        md_path.write_text(_markdown_report(payload))
        print(f"\nWrote {json_path}\nWrote {md_path}")
        return 0

    rows = _score_paired(
        scenarios, candidate=args.candidate, baseline=args.baseline, opponent=args.opponent, horizon_s=args.horizon
    )
    aggregates = _aggregate_by_family(rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "git_revision": _git_revision(),
        "candidate": args.candidate,
        "baseline": args.baseline,
        "opponent": args.opponent,
        "horizon_s": args.horizon,
        "n_scenarios": len(scenarios),
        "aggregates": aggregates,
        "results": rows,
        "harvest_report": harvest_report,
    }

    json_path = args.output_dir / f"scenario_bench_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = args.output_dir / f"scenario_bench_{timestamp}.md"
    md_path.write_text(_markdown_report(payload))

    print(f"\nWrote {json_path}\nWrote {md_path}")
    overall_mean = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"Overall mean delta: {overall_mean:+.2f} over {len(rows)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
