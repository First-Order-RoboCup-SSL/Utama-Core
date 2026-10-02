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
  - Hand-authored anchors (`utama_core.scenario_bench.hand_authored_scenarios`) —
    always available, no dependency on any replay data.
  - Harvested restart-transition scenarios (`--harvest-from RUN_DIR`) — only
    from a tagged, trustworthy tournament run (`.stats.json` with zero
    `stall_events`, see `utama_core.scenario_bench.scenario_harvester`'s module
    docstring for why this gate exists). NOT run by this tool — point it at
    an already-completed `round_robin.py` run directory. `--open-play N`
    adds up to N open-play starts of each kind per match (a pass about to be
    made, a ball just lost).

For each scenario, both `--candidate` and `--baseline` play the SAME
scenario against the SAME `--opponent` (paired comparison, per item 14's
"Compute discipline" section — the variance of the *difference* is what
matters, not either absolute score). The report is the per-scenario and
per-family paired outcome delta, never an absolute score.

rsim is deterministic, so a scenario can also be played from `--repeats` starts
(`start.jittered`: robots away from the ball nudged a few cm; seed 0
is the scenario as authored), and both sides get the same starts. The default
is 1: more scenarios detect a change better than more repeats of each (every
start from one round-robin, once each, finds passes aimed 10 degrees off at
t = -7.6). The standard error of the mean delta is across scenarios.

The baseline is either a `--baseline` config run in this process, or
`--against-results FILE`: the candidate outcomes of an earlier run's JSON. The
second is how to A/B a code change (shared tactics, planner): run the bench at
commit A, then at commit B with `--against-results` pointing at A's JSON. The
opponent, horizon and repeats must match. With neither, it only records the
candidate's outcomes, e.g. to serve as a later run's `--against-results`.

Scenario outcomes count real ball losses and flag stalls; see
`utama_core.scenario_bench.scenario_scorer`.

`--save-bank PATH` persists the currently-loaded scenario set (hand-authored
+ optional `--harvest-from`, after `--families` filtering) to a single JSON
file via `start.save_bank` — a few KB even for hundreds of
scenarios, since a scenario is field state only, no trajectories. `--load-bank
PATH` loads scenarios from a previously-saved bank instead of hand-authored/
`--harvest-from` (the two are mutually exclusive as *sources*: a loaded bank
is meant to be the frozen set a prior save produced, not a starting point to
silently merge fresh sources into — see item 14's "Immutable per bank version"
note). A bank is every harvested start, near-duplicates dropped; there is no
play-forward screen (one was tried: it kept starts where the OPPONENT changed
the outcome, and threw away as many starts where the candidate did). To grow a
bank from a new round-robin, `--merge-into` the current bank: harvested
scenarios that duplicate it are dropped, and the new bank is the old one plus
the rest. `--workers N` scores N scenarios at a time.

`--reuse` takes a start's outcome from `replays/match_cache/` instead of playing it
when nothing that start runs has changed since it was stored
(`utama_core.replay.fingerprint.bench_key`: the jittered start, both configs' code, the
shared code and environment, the horizon). A baseline config whose code is unchanged
then costs nothing, and only the candidate plays. `--spot-check F` (default 0.05) also
replays that fraction of the reusable starts and compares them with their records; a
difference means the fingerprint missed a dependency, is printed loudly, evicts the
records this run used, and makes the run exit non-zero.
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
        --save-bank utama_core/scenario_bench/banks/bank_v5.json --bank-id bank_v5 --list-scenarios

    # Grow a bank from a new round-robin (harvest, drop duplicates, save):
    pixi run python tools/scenario_bench.py --harvest-from replays/tournament_<id> --open-play 2 \\
        --merge-into utama_core/scenario_bench/banks/bank_v5.json --save-bank utama_core/scenario_bench/banks/bank_v6.json \\
        --list-scenarios

    # A/B a code change: record at commit A, compare at commit B.
    pixi run python tools/scenario_bench.py --load-bank bank_v5.json \\
        --candidate build_tiki_taka_kernel_strategy --opponent build_low_block_kernel_strategy
    pixi run python tools/scenario_bench.py --load-bank bank_v5.json \\
        --candidate build_tiki_taka_kernel_strategy --opponent build_low_block_kernel_strategy \\
        --against-results scenario_bench_results/scenario_bench_<commit A>.json

    # Score against the frozen bank later, without re-harvesting:
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v5.json \\
        --candidate build_tiki_taka_kernel_strategy --baseline build_default_kernel_strategy \\
        --opponent build_low_block_kernel_strategy
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.replay import match_cache
from utama_core.replay.fingerprint import CodeGraph, bench_key
from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    enable_sim_reuse,
)
from utama_core.scenario_bench.hand_authored_scenarios import (
    all_hand_authored_scenarios,
)
from utama_core.scenario_bench.scenario_harvester import harvest_run_dir
from utama_core.scenario_bench.scenario_scorer import (
    _resolve_config_name,
    score_scenario,
)
from utama_core.scenario_bench.start import (
    BenchScenario,
    drop_near_duplicates,
    jittered,
    load_bank,
    save_bank,
)

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
    """(scenarios to score, the `--merge-into` bank's scenarios, harvest report)."""
    harvest_report: dict = {}
    merged: list[BenchScenario] = []

    if args.load_bank is not None:
        # A persisted bank replaces the hand-authored + harvest sources
        # entirely rather than merging with them — it's meant to be the
        # frozen set a prior `--save-bank` produced (see
        # `start.save_bank`'s "immutable per bank version" note),
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
    if args.harvest_from is not None:
        before = len(scenarios)
        scenarios = drop_near_duplicates(scenarios, keep=merged)
        against = f"{args.merge_into} or each other" if args.merge_into else "each other"
        print(f"{before - len(scenarios)} of {before} scenarios duplicate {against}; dropped")

    return scenarios, merged, harvest_report


def _save(args: argparse.Namespace, scenarios: list[BenchScenario]) -> None:
    if args.save_bank is not None:
        bank_id = args.bank_id or args.save_bank.stem
        save_bank(scenarios, args.save_bank, bank_id=bank_id)
        print(f"Saved {len(scenarios)} scenarios to {args.save_bank} (bank_id={bank_id})")


def _start_result(bench_scenario: BenchScenario, config: str, opponent: str, horizon_s: float) -> dict:
    r = score_scenario(bench_scenario, candidate_config=config, opponent_config=opponent, horizon_s=horizon_s)
    return {"outcome": int(r.outcome), "foul": bool(r.foul), "stalled": bool(r.stalled), "error": r.error}


def _runs(
    bench_scenario: BenchScenario,
    config: str,
    opponent: str,
    horizon_s: float,
    repeats: int,
    plan: Optional[dict] = None,
    fresh: Optional[dict] = None,
) -> dict:
    """`config` vs `opponent` from `repeats` jittered starts (seed 0 = as authored). With
    `--reuse`, `plan` maps `(config, seed)` to `(key, stored record or None)`: a start with
    a record is not played, and every start played goes into `fresh` under its key."""
    results = []
    for seed in range(repeats):
        key, record = (plan or {}).get((config, seed), (None, None))
        if record is None:
            record = {"result": _start_result(jittered(bench_scenario, seed), config, opponent, horizon_s)}
            if key is not None and fresh is not None:
                fresh[key] = record
        results.append(record["result"])
    return {
        "outcomes": [r["outcome"] for r in results],
        "fouls": sum(r["foul"] for r in results),
        "stalls": sum(r["stalled"] for r in results),
        "errors": [r["error"] for r in results if r["error"]],
    }


def _load_against(path: Path, *, opponent: str, horizon_s: float, repeats: int) -> tuple[dict[str, list[int]], dict]:
    """Baseline outcomes from an earlier run's JSON (typically another commit), keyed by
    scenario id. Refuses a file scored against a different opponent/horizon/repeats:
    those outcomes are not comparable."""
    payload = json.loads(path.read_text())
    for key, want in (("opponent", opponent), ("horizon_s", horizon_s), ("repeats", repeats)):
        if payload.get(key) != want:
            raise SystemExit(f"--against-results {path}: {key} is {payload.get(key)!r}, this run uses {want!r}")
    outcomes = {
        row["scenario_id"]: row["candidate_outcomes"] for row in payload["results"] if not row.get("candidate_errors")
    }
    source = {"path": str(path), "git_revision": payload.get("git_revision"), "candidate": payload.get("candidate")}
    return outcomes, source


def _score_one(
    item: tuple[BenchScenario, Optional[dict]],
    *,
    candidate: str,
    opponent: str,
    horizon_s: float,
    repeats: int,
    baseline,
) -> tuple[dict, Optional[dict], dict]:
    bench_scenario, plan = item
    enable_sim_reuse()  # a worker plays many starts: keep its sim subprocess between them
    fresh: dict = {}
    cand = _runs(bench_scenario, candidate, opponent, horizon_s, repeats, plan, fresh)
    base = _runs(bench_scenario, baseline, opponent, horizon_s, repeats, plan, fresh) if baseline else None
    return cand, base, fresh


class _Reuse:
    """`--reuse`'s bookkeeping for one bench run: every start's key, the records it may
    reuse, the ones it replays to check, and what it played."""

    def __init__(self, scenarios, configs, opponent: str, horizon_s: float, repeats: int, spot_check: float):
        self.args = (scenarios, configs, opponent, horizon_s, repeats)
        self.cache = match_cache.MatchCache()
        self.keys = self._keys()
        self.stored = {k: rec for k in self.keys.values() if (rec := self.cache.get(k)) is not None}
        self.spot_checked = match_cache.spot_check_sample(self.stored, spot_check)
        self.used: set[str] = set()  # reused records, for eviction after a mismatch
        self.fresh: dict[str, dict] = {}

    def _keys(self) -> dict[tuple[str, str, int], str]:
        scenarios, configs, opponent, horizon_s, repeats = self.args
        graph = CodeGraph()
        return {
            (bs.scenario_id, config, seed): bench_key(
                graph,
                jittered(bs, seed).to_dict(),
                _resolve_config_name(config),
                _resolve_config_name(opponent),
                horizon_s=horizon_s,
            )
            for bs in scenarios
            for config in configs
            for seed in range(repeats)
        }

    def plan(self, bench_scenario: BenchScenario) -> dict:
        out = {}
        for (sid, config, seed), key in self.keys.items():
            if sid != bench_scenario.scenario_id:
                continue
            record = None if key in self.spot_checked else self.stored.get(key)
            out[(config, seed)] = (key, record)
            if record is not None:
                self.used.add(key)
        return out

    def finish(self) -> dict:
        """Compare the spot-checks, store what was played, and report counts."""
        mismatches = [
            k for k in self.spot_checked if k in self.fresh and match_cache.differences(self.stored[k], self.fresh[k])
        ]
        if mismatches:
            for k in mismatches + sorted(self.used):
                self.cache.evict(k)
            print(
                f"\n*** --reuse SPOT-CHECK MISMATCH: {len(mismatches)} replayed start(s) differ from their stored "
                f"records. The fingerprint missed something they depend on; those records and the {len(self.used)} "
                "this run reused are evicted, and its result can't be trusted: rerun without --reuse. ***",
                flush=True,
            )
        elif self._keys() != self.keys:
            print("\n--reuse: the code changed while this run played; storing nothing", flush=True)
        else:
            for k, record in self.fresh.items():
                if not record["result"]["error"] and k not in self.spot_checked:
                    self.cache.put(k, record)
        checked = len([k for k in self.spot_checked if k in self.fresh])
        print(
            f"--reuse: {len(self.used)} start(s) reused, {len(self.fresh)} played, spot-check "
            f"{checked - len(mismatches)}/{checked} identical",
            flush=True,
        )
        return {
            "reused": len(self.used),
            "played": len(self.fresh),
            "spot_checked": checked,
            "mismatches": len(mismatches),
        }


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
    stop_at_t: Optional[float] = None,
    check_every: int = 100,
    reuse: Optional[_Reuse] = None,
) -> tuple[list[dict], Optional[str]]:
    """Candidate outcomes per scenario, paired with the baseline's on the same jittered
    starts: a `baseline` config run here, or `against` outcomes from an earlier run.

    With `stop_at_t`, scenarios are scored in a shuffled order (a bank is ordered by match
    and family, so its first starts are not a fair sample), `check_every` at a time, and
    scoring stops at a check where `_stop_reason` gives one; that reason is returned with
    the rows (None when every scenario was scored)."""
    rows = []
    total = len(scenarios)
    play = functools.partial(
        _score_one, candidate=candidate, opponent=opponent, horizon_s=horizon_s, repeats=repeats, baseline=baseline
    )
    if stop_at_t is not None:
        scenarios = random.Random(0).sample(scenarios, len(scenarios))
        batches = [scenarios[i : i + check_every] for i in range(0, len(scenarios), check_every)]
    else:
        batches = [scenarios]
    for batch in batches:
        items = [(bs, reuse.plan(bs) if reuse else None) for bs in batch]

        def results(items=items):
            # a generator, so `_score_batch` prints each start as it comes back
            for cand, base, fresh in _parallel_map(play, items, workers):
                if reuse:
                    reuse.fresh.update(fresh)
                yield cand, base

        _score_batch(rows, batch, results(), against, total)
        if stop_at_t is not None and len(rows) < total:
            deltas = [r["delta"] for r in rows if r["delta"] is not None]
            stopped = _stop_reason(deltas, stop_at_t)
            if stopped:
                print(f"Stopped after {len(rows)} of {total} scenarios ({stopped}): {_overall(rows)}", flush=True)
                return rows, stopped
    return rows, None


# Futility: stop once the mean delta is confidently below FUTILE_BELOW, i.e. its upper
# bound |mean| + FUTILE_Z * stderr is. 0.15 is about the smallest mean delta a full
# bank_v5 pass detects at |t| >= 4 (4 * 1.02 / sqrt(846) = 0.14, 1.02 the spread of
# per-start deltas), so a candidate stopped as futile would not have been detected by
# the full pass either. 2.5 is a one-sided 5% bound split over the 8 checks of a bank_v5
# pass (0.05 / 8). Calibration: docs/STRATEGY_DEVELOPMENT.md, "--stop-at-t".
FUTILE_BELOW = 0.15
FUTILE_Z = 2.5


def _stop_reason(deltas: list[float], stop_at_t: float) -> Optional[str]:
    """Why to stop scoring: "detected" once |t| of `deltas` reaches `stop_at_t`, "futile"
    once the difference is confidently smaller than FUTILE_BELOW, else None."""
    if len(deltas) < 2:
        return None
    if abs(_t(deltas)) >= stop_at_t:
        return "detected"
    stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
    if abs(statistics.fmean(deltas)) + FUTILE_Z * stderr < FUTILE_BELOW:
        return "futile"
    return None


def _t(deltas: list[float]) -> float:
    """mean / stderr of `deltas`; infinite when every delta is the same nonzero value."""
    if len(deltas) < 2:
        return 0.0
    mean = statistics.fmean(deltas)
    stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
    if stderr == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return mean / stderr


def _score_batch(rows: list[dict], batch: list[BenchScenario], results: Iterable, against, total: int) -> None:
    for bench_scenario, (cand, base) in zip(batch, results):
        index = len(rows) + 1
        sid = bench_scenario.scenario_id
        print(f"[{index}/{total}] {sid} ({bench_scenario.provenance.family.value})", flush=True)
        base_outcomes = base["outcomes"] if base else (against or {}).get(sid)
        delta = None
        # a run that errored (e.g. the start could not be set up) is not an outcome
        if base_outcomes is not None and not cand["errors"] and not (base and base["errors"]):
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
        f"Scenarios scored: {payload.get('n_scored', payload['n_scenarios'])} of {payload['n_scenarios']}"
        + (f" (stopped early: {payload['stopped']})" if payload.get("stopped") else ""),
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
        default=1,
        help="jittered starts per scenario (seed 0 = as authored; default 1: more scenarios beat more repeats)",
    )
    parser.add_argument(
        "--stop-at-t",
        type=float,
        default=None,
        help="score a shuffled sample 100 scenarios at a time and stop once |t| reaches this (4 is a safe choice: "
        "t is looked at repeatedly, so 2 would give false alarms), or once the mean delta is confidently under "
        f"{FUTILE_BELOW} (futile); for screening candidates, not for a baseline",
    )
    parser.add_argument("--opponent", help="opponent strategy config both candidate and baseline play against")
    parser.add_argument("--horizon", type=float, default=20.0, help="sim seconds ticked per scenario (default 20)")
    parser.add_argument(
        "--harvest-from",
        type=Path,
        default=None,
        help="a completed round_robin.py run dir to harvest restart scenarios from",
    )
    parser.add_argument(
        "--open-play",
        type=int,
        default=0,
        help="with --harvest-from, also harvest up to this many open-play scenarios of each kind per match "
        "(see scenario_harvester.find_open_play_events; default 0)",
    )
    parser.add_argument("--families", nargs="+", default=None, help="restrict to these ScenarioFamily values")
    parser.add_argument("--list-scenarios", action="store_true", help="print the loaded bank and exit")
    parser.add_argument(
        "--load-bank",
        type=Path,
        default=None,
        help="load scenarios from a persisted bank JSON (see start.save_bank) "
        "instead of hand-authored/--harvest-from",
    )
    parser.add_argument(
        "--save-bank",
        type=Path,
        default=None,
        help="write the loaded scenario set (after --families filtering) to this path as a persisted bank JSON",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="an existing bank: drop harvested scenarios that duplicate it (start.is_near_duplicate), "
        "and --save-bank writes it plus the new ones, as a new bank",
    )
    parser.add_argument("--workers", type=int, default=1, help="processes to score scenarios on (default 1)")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="take each start's outcome from replays/match_cache/ when the code it runs is unchanged (see above)",
    )
    parser.add_argument(
        "--spot-check",
        type=float,
        default=0.05,
        help="with --reuse, the fraction of reusable starts to replay and compare (default 0.05, at least one)",
    )
    parser.add_argument(
        "--bank-id",
        default=None,
        help="bank_id to record when using --save-bank (default: the output filename's stem)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scenario_bench_results"))
    args = parser.parse_args()

    if not args.list_scenarios:
        missing = [name for name in ("candidate", "opponent") if getattr(args, name) is None]
        if missing:
            parser.error(f"--{', --'.join(missing)} required unless --list-scenarios")
        if args.baseline and args.against_results:
            parser.error("--baseline and --against-results are alternatives")
    if args.merge_into is not None and (args.harvest_from is None or args.save_bank is None):
        parser.error("--merge-into needs --harvest-from and --save-bank")

    return args


def main() -> int:
    args = parse_args()
    scenarios, merged, harvest_report = _load_bank(args)
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

    against, against_source = None, None
    if args.against_results:
        against, against_source = _load_against(
            args.against_results, opponent=args.opponent, horizon_s=args.horizon, repeats=args.repeats
        )
    configs = [args.candidate] + ([args.baseline] if args.baseline else [])
    reuse = (
        _Reuse(scenarios, configs, args.opponent, args.horizon, args.repeats, args.spot_check) if args.reuse else None
    )
    rows, stopped = _score(
        scenarios,
        candidate=args.candidate,
        opponent=args.opponent,
        horizon_s=args.horizon,
        repeats=args.repeats,
        baseline=args.baseline,
        against=against,
        workers=args.workers,
        stop_at_t=args.stop_at_t,
        reuse=reuse,
    )
    reuse_report = reuse.finish() if reuse else None
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
        "n_scored": len(rows),
        "stop_at_t": args.stop_at_t,
        # why scoring ended early: "detected" (|t| reached stop_at_t) or "futile" (see
        # FUTILE_BELOW); None when every scenario was scored
        "stopped": stopped,
        "reuse": reuse_report,
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
        print(_overall(rows) + (f" (stopped early: {stopped})" if stopped else ""))
    return 1 if reuse_report and reuse_report["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
