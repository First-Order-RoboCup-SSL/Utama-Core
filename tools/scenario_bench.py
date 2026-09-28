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
    an already-completed `smoke_tournament.py` run directory. `--open-play N`
    adds up to N open-play starts of each kind per match (a pass about to be
    made, a ball just lost).

For each scenario, both `--candidate` and `--baseline` play the SAME
scenario against the SAME `--opponent` (paired comparison, per item 14's
"Compute discipline" section — the variance of the *difference* is what
matters, not either absolute score). The report is the per-scenario and
per-family paired outcome delta, never an absolute score.

rsim is deterministic, so each scenario is played from `--repeats` starts
(`bench_scenario.jittered`: robots away from the ball nudged a few cm; seed 0
is the scenario as authored), and both sides get the same starts. The report
gives the candidate's own spread across starts (seed noise) and the standard
error of the mean delta across scenarios.

The baseline is either a `--baseline` config run in this process, or
`--against-results FILE`: the candidate outcomes of an earlier run's JSON. The
second is how to A/B a code change (shared tactics, planner): run the bench at
commit A, then at commit B with `--against-results` pointing at A's JSON. The
opponent, horizon and repeats must match. With neither, it only records the
candidate's outcomes, e.g. to serve as a later run's `--against-results`.

Scenario outcomes count real ball losses and flag stalls; see
`utama_core.replay.scenario_scorer`.

`--save-bank PATH` persists the currently-loaded scenario set (hand-authored
+ optional `--harvest-from`, after `--families` filtering) to a single JSON
file via `bench_scenario.save_bank` — a few KB even for hundreds of
scenarios, since a scenario is field state only, no trajectories. `--load-bank
PATH` loads scenarios from a previously-saved bank instead of hand-authored/
`--harvest-from` (the two are mutually exclusive as *sources*: a loaded bank
is meant to be the frozen, already-screened set a prior save produced, not a
starting point to silently merge fresh sources into — see item 14's
"Immutable per bank version" note). With `--dynamic-screen`, `--save-bank`
keeps only the informative and noisy scenarios. To grow a bank from a new
round-robin, `--merge-into` the current bank: harvested scenarios that
duplicate it are dropped before screening, and the new bank is the old one
plus the survivors. `--workers N` screens/scores N scenarios at a time.
Still not built: lifecycle promotion (candidate -> validated -> active) or a
ladder (slow half).

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

    # Grow a bank from a new round-robin (harvest, drop duplicates, screen, save survivors):
    pixi run python tools/scenario_bench.py --harvest-from replays/tournament_<id> --open-play 2 \\
        --merge-into utama_core/replay/banks/bank_v5.json --save-bank utama_core/replay/banks/bank_v6.json \\
        --dynamic-screen --workers 15

    # A/B a code change: record at commit A, compare at commit B.
    pixi run python tools/scenario_bench.py --load-bank bank_v1.json \\
        --candidate build_tiki_taka_kernel_strategy --opponent build_low_block_kernel_strategy
    pixi run python tools/scenario_bench.py --load-bank bank_v1.json \\
        --candidate build_tiki_taka_kernel_strategy --opponent build_low_block_kernel_strategy \\
        --against-results scenario_bench_results/scenario_bench_<commit A>.json

    # Score against the frozen bank later, without re-harvesting:
    pixi run python tools/scenario_bench.py --load-bank utama_core/replay/banks/bank_v1.json \\
        --candidate build_tiki_taka_kernel_strategy --baseline build_default_kernel_strategy \\
        --opponent build_low_block_kernel_strategy
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.replay.bench_scenario import (
    BenchScenario,
    drop_near_duplicates,
    jittered,
    load_bank,
    save_bank,
)
from utama_core.replay.dynamic_screen import (
    DEFAULT_SCREEN_POOL,
    ScreenVerdict,
    screen_scenario,
)
from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios
from utama_core.replay.scenario_harvester import harvest_run_dir
from utama_core.replay.scenario_scorer import score_scenario

SCHEMA_VERSION = 1


def _git_revision() -> str | None:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _parallel_map(fn: Callable, items: Iterable, workers: int) -> Iterator:
    """`map`, over `workers` processes when more than one (results stay in order)."""
    if workers <= 1:
        yield from map(fn, items)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(fn, items)


def _load_bank(args: argparse.Namespace) -> tuple[list[BenchScenario], list[BenchScenario], dict]:
    """(scenarios to screen/score, the `--merge-into` bank's scenarios, harvest report)."""
    harvest_report: dict = {}
    merged: list[BenchScenario] = []

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
                open_play_per_match=args.open_play,
            )
            scenarios.extend(harvested)

    if args.families:
        wanted = set(args.families)
        scenarios = [s for s in scenarios if s.provenance.family.value in wanted]

    if args.merge_into is not None:
        _bank_id, merged = load_bank(args.merge_into)
        before = len(scenarios)
        scenarios = drop_near_duplicates(scenarios, keep=merged)
        print(f"{before - len(scenarios)} of {before} scenarios duplicate {args.merge_into} or each other; dropped")

    return scenarios, merged, harvest_report


def _save(args: argparse.Namespace, scenarios: list[BenchScenario]) -> None:
    if args.save_bank is not None:
        bank_id = args.bank_id or args.save_bank.stem
        save_bank(scenarios, args.save_bank, bank_id=bank_id)
        print(f"Saved {len(scenarios)} scenarios to {args.save_bank} (bank_id={bank_id})")


def _screen_one(bench_scenario: BenchScenario, *, horizon_s: float, repeats: int) -> dict:
    result = screen_scenario(
        bench_scenario,
        champion_config="build_default_kernel_strategy",
        pool_configs=DEFAULT_SCREEN_POOL,
        horizon_s=horizon_s,
        repeats=repeats,
    )
    return {
        "scenario_id": result.scenario_id,
        "verdict": result.verdict.value,
        "outcomes": list(result.outcomes),
        "outcome_stdev": result.outcome_stdev,
        "seed_stdev": result.seed_stdev,
        "policy_stdev": result.policy_stdev,
    }


def _run_dynamic_screen(
    scenarios: list[BenchScenario], *, horizon_s: float, repeats: int, workers: int = 1
) -> list[dict]:
    screen = functools.partial(_screen_one, horizon_s=horizon_s, repeats=repeats)
    results = []
    for index, result in enumerate(_parallel_map(screen, scenarios, workers), start=1):
        print(f"[{index}/{len(scenarios)}] {result['scenario_id']}: {result['verdict']}", flush=True)
        results.append(result)
    return results


def _runs(bench_scenario: BenchScenario, config: str, opponent: str, horizon_s: float, repeats: int) -> dict:
    """`config` vs `opponent` from `repeats` jittered starts (seed 0 = as authored)."""
    results = [
        score_scenario(
            jittered(bench_scenario, seed), candidate_config=config, opponent_config=opponent, horizon_s=horizon_s
        )
        for seed in range(repeats)
    ]
    return {
        "outcomes": [int(r.outcome) for r in results],
        "fouls": sum(r.foul for r in results),
        "stalls": sum(r.stalled for r in results),
        "errors": [r.error for r in results if r.error],
    }


def _load_against(path: Path, *, opponent: str, horizon_s: float, repeats: int) -> tuple[dict[str, list[int]], dict]:
    """Baseline outcomes from an earlier run's JSON (typically another commit), keyed by
    scenario id. Refuses a file scored against a different opponent/horizon/repeats:
    those outcomes are not comparable."""
    payload = json.loads(path.read_text())
    for key, want in (("opponent", opponent), ("horizon_s", horizon_s), ("repeats", repeats)):
        if payload.get(key) != want:
            raise SystemExit(f"--against-results {path}: {key} is {payload.get(key)!r}, this run uses {want!r}")
    outcomes = {row["scenario_id"]: row["candidate_outcomes"] for row in payload["results"]}
    source = {"path": str(path), "git_revision": payload.get("git_revision"), "candidate": payload.get("candidate")}
    return outcomes, source


def _score_one(
    bench_scenario: BenchScenario, *, candidate: str, opponent: str, horizon_s: float, repeats: int, baseline
) -> tuple[dict, Optional[dict]]:
    cand = _runs(bench_scenario, candidate, opponent, horizon_s, repeats)
    base = _runs(bench_scenario, baseline, opponent, horizon_s, repeats) if baseline else None
    return cand, base


def _score(
    scenarios: list[BenchScenario],
    *,
    candidate: str,
    opponent: str,
    horizon_s: float,
    repeats: int,
    baseline: Optional[str] = None,
    against: Optional[dict[str, list[int]]] = None,
    workers: int = 1,
) -> list[dict]:
    """Candidate outcomes per scenario, paired with the baseline's on the same jittered
    starts: a `baseline` config run here, or `against` outcomes from an earlier run."""
    rows = []
    total = len(scenarios)
    play = functools.partial(
        _score_one, candidate=candidate, opponent=opponent, horizon_s=horizon_s, repeats=repeats, baseline=baseline
    )
    for index, (bench_scenario, (cand, base)) in enumerate(
        zip(scenarios, _parallel_map(play, scenarios, workers)), start=1
    ):
        sid = bench_scenario.scenario_id
        print(f"[{index}/{total}] {sid} ({bench_scenario.provenance.family.value})", flush=True)
        base_outcomes = base["outcomes"] if base else (against or {}).get(sid)
        delta = None
        if base_outcomes is not None:
            delta = statistics.fmean(cand["outcomes"]) - statistics.fmean(base_outcomes)
        print(
            f"  candidate={cand['outcomes']} baseline={base_outcomes}"
            + (f" delta={delta:+.2f}" if delta is not None else "")
            + (f" stalls={cand['stalls']}" if cand["stalls"] else ""),
            flush=True,
        )
        rows.append(
            {
                "scenario_id": sid,
                "family": bench_scenario.provenance.family.value,
                "trigger": bench_scenario.provenance.trigger.value,
                "perspective": bench_scenario.provenance.perspective,
                "candidate_outcomes": cand["outcomes"],
                "baseline_outcomes": base_outcomes,
                "delta": delta,
                "candidate_fouls": cand["fouls"],
                "candidate_stalls": cand["stalls"],
                "baseline_fouls": base["fouls"] if base else None,
                "baseline_stalls": base["stalls"] if base else None,
                "candidate_errors": cand["errors"],
                "baseline_errors": base["errors"] if base else [],
            }
        )
    return rows


def _aggregate_by_family(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)

    aggregates = []
    for family, family_rows in grouped.items():
        deltas = [r["delta"] for r in family_rows if r["delta"] is not None]
        n = len(deltas)
        aggregates.append(
            {
                "family": family,
                "n_scenarios": len(family_rows),
                "n_paired": n,
                "mean_delta": statistics.fmean(deltas) if n else None,
                # standard error of the mean delta across scenarios: a mean delta within
                # about two of these of zero is not a difference
                "stderr": statistics.stdev(deltas) / math.sqrt(n) if n > 1 else None,
                "wins": sum(1 for d in deltas if d > 0),
                "losses": sum(1 for d in deltas if d < 0),
                "ties": sum(1 for d in deltas if d == 0),
                # mean over scenarios of the candidate's outcome stdev across jittered starts
                "seed_noise": statistics.fmean(statistics.pstdev(r["candidate_outcomes"]) for r in family_rows),
                "candidate_stalls": sum(r["candidate_stalls"] for r in family_rows),
            }
        )
    return sorted(aggregates, key=lambda a: a["family"])


def _overall(rows: list[dict]) -> Optional[str]:
    """One line: mean delta over scenarios, its standard error, and t = mean / stderr.
    |t| under about 2 is within chance; the sign says which side was better."""
    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    if not deltas:
        return None
    mean = statistics.fmean(deltas)
    if len(deltas) < 2:
        return f"Overall mean delta {mean:+.3f} over 1 scenario"
    stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
    if stderr == 0:
        return f"Overall mean delta {mean:+.3f}, the same on all {len(deltas)} scenarios"
    return (
        f"Overall mean delta {mean:+.3f} (stderr {stderr:.3f}, t {mean / stderr:+.2f}) over {len(deltas)} "
        f"scenarios; |t| under ~2 is within chance"
    )


def _fmt(x: Optional[float], spec: str = "+.2f") -> str:
    return "-" if x is None else format(x, spec)


def _markdown_report(payload: dict) -> str:
    baseline = payload["baseline"] or (
        f"results of `{payload['against']['candidate']}` at `{payload['against']['git_revision']}` "
        f"({payload['against']['path']})"
        if payload.get("against")
        else None
    )
    lines = [
        "# Scenario bench report",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Git revision: `{payload['git_revision'] or 'unknown'}`  ",
        f"Candidate: `{payload['candidate']}`  ",
        f"Baseline: {baseline or 'none'}  ",
        f"Opponent: `{payload['opponent']}`  ",
        f"Horizon: {payload['horizon_s']}s, {payload.get('repeats', 1)} jittered starts per scenario  ",
        f"Scenarios scored: {payload['n_scenarios']}",
        "",
        "Delta is mean candidate outcome minus mean baseline outcome over the same jittered starts, "
        "on the ordinal scale (GOAL_AGAINST=-3 ... NEUTRAL=0 ... GOAL_FOR=+3), same opponent. "
        "`stderr` is across scenarios; `seed noise` is the candidate's own spread across starts. "
        "Stalls are counted, not ranked. "
        "This is a proxy signal, not an acceptance gate — see roadmap item 14's Goodhart guard.",
        "",
        *([f"**{_overall(payload['results'])}**", ""] if _overall(payload["results"]) else []),
        "## By family",
        "",
        "| Family | N | Mean delta | Stderr | Wins | Losses | Ties | Seed noise | Stalls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agg in payload["aggregates"]:
        lines.append(
            f"| {agg['family']} | {agg['n_scenarios']} | {_fmt(agg['mean_delta'])} | {_fmt(agg['stderr'], '.2f')} | "
            f"{agg['wins']} | {agg['losses']} | {agg['ties']} | {agg['seed_noise']:.2f} | {agg['candidate_stalls']} |"
        )

    lines.extend(
        [
            "",
            "## Per-scenario",
            "",
            "| Scenario | Family | Candidate | Baseline | Delta | Stalls |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            f"| {row['scenario_id']} | {row['family']} | {row['candidate_outcomes']} | "
            f"{row['baseline_outcomes'] if row['baseline_outcomes'] is not None else '-'} | {_fmt(row['delta'])} | "
            f"{row['candidate_stalls']} |"
        )

    errors = [r for r in payload["results"] if r["candidate_errors"] or r["baseline_errors"]]
    lines.extend(["", "## Errors", ""])
    if not errors:
        lines.append("None.")
    else:
        for row in errors:
            lines.append(
                f"- `{row['scenario_id']}`: candidate={row['candidate_errors']} baseline={row['baseline_errors']}"
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
        lines.extend(
            [
                "",
                "## Dynamic screen",
                "",
                "| Scenario | Verdict | Outcomes | Stdev | Seed stdev | Policy stdev |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in payload["dynamic_screen"]:
            lines.append(
                f"| {row['scenario_id']} | {row['verdict']} | {row['outcomes']} | {row['outcome_stdev']:.2f} | "
                f"{row['seed_stdev']:.2f} | {row['policy_stdev']:.2f} |"
            )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", help="candidate strategy config, e.g. build_tiki_taka_kernel_strategy")
    parser.add_argument("--baseline", help="baseline strategy config to diff against, run in this process")
    parser.add_argument(
        "--against-results",
        type=Path,
        default=None,
        help="diff against the candidate outcomes in an earlier run's JSON instead of a --baseline config "
        "(A/B across commits: run once at each commit, pass the first run's JSON to the second)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="jittered starts per scenario (seed 0 = as authored; default 3)",
    )
    parser.add_argument("--opponent", help="opponent strategy config both candidate and baseline play against")
    parser.add_argument("--horizon", type=float, default=20.0, help="sim seconds ticked per scenario (default 20)")
    parser.add_argument(
        "--harvest-from",
        type=Path,
        default=None,
        help="a completed smoke_tournament.py run dir to harvest restart scenarios from",
    )
    parser.add_argument(
        "--open-play",
        type=int,
        default=0,
        help="with --harvest-from, also harvest up to this many open-play scenarios of each kind per match "
        "(see scenario_harvester.find_open_play_events; default 0)",
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
        help="write the loaded scenario set (after --families filtering) to this path as a persisted bank JSON; "
        "with --dynamic-screen, only the informative and noisy scenarios",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="an existing bank: drop harvested scenarios that duplicate it (bench_scenario.is_near_duplicate) "
        "before screening, and --save-bank writes it plus the new survivors, as a new bank",
    )
    parser.add_argument("--workers", type=int, default=1, help="processes to screen/score scenarios on (default 1)")
    parser.add_argument(
        "--bank-id",
        default=None,
        help="bank_id to record when using --save-bank (default: the output filename's stem)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scenario_bench_results"))
    args = parser.parse_args()

    if not args.list_scenarios and not args.dynamic_screen:
        missing = [name for name in ("candidate", "opponent") if getattr(args, name) is None]
        if missing:
            parser.error(f"--{', --'.join(missing)} required unless --list-scenarios or --dynamic-screen")
        if args.baseline and args.against_results:
            parser.error("--baseline and --against-results are alternatives")
    if args.merge_into is not None and (args.harvest_from is None or args.save_bank is None):
        parser.error("--merge-into needs --harvest-from and --save-bank")

    return args


def main() -> int:
    args = parse_args()
    scenarios, merged, harvest_report = _load_bank(args)
    if not args.dynamic_screen:
        _save(args, merged + scenarios)

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
        screen_results = _run_dynamic_screen(
            scenarios, horizon_s=args.horizon, repeats=args.repeats, workers=args.workers
        )
        kept = {
            r["scenario_id"]
            for r in screen_results
            if r["verdict"] in (ScreenVerdict.INFORMATIVE.value, ScreenVerdict.NOISY.value)
        }
        _save(args, merged + [s for s in scenarios if s.scenario_id in kept])
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
            "repeats": args.repeats,
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

    against, against_source = None, None
    if args.against_results:
        against, against_source = _load_against(
            args.against_results, opponent=args.opponent, horizon_s=args.horizon, repeats=args.repeats
        )
    rows = _score(
        scenarios,
        candidate=args.candidate,
        opponent=args.opponent,
        horizon_s=args.horizon,
        repeats=args.repeats,
        baseline=args.baseline,
        against=against,
        workers=args.workers,
    )
    aggregates = _aggregate_by_family(rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "git_revision": _git_revision(),
        "candidate": args.candidate,
        "baseline": args.baseline,
        "against": against_source,
        "opponent": args.opponent,
        "horizon_s": args.horizon,
        "repeats": args.repeats,
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
    if _overall(rows):
        print(_overall(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
