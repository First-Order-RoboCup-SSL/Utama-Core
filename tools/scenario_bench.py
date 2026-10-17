#!/usr/bin/env python3
# ruff: noqa: E402
"""Situation bench: how a candidate strategy changes goals per match, estimated from short
paired replays of the situations goals come from.

    effect = sum over situation types of (how often it happens per match)
                                          x (candidate's P(goal) from it - baseline's)

Each start of the bank is a situation of the goal ledger (`utama_core.analysis.goal_ledger`)
from the candidate's view: own or opponent ball won or free kick in each third, own or
opponent kickoff or penalty, force start. Its outcome is the first goal within `--horizon`
(default 10 s, the ledger's window): for, against, or none. P(for) and P(against) per type
are means over the type's starts, and the change is paired: the candidate and the baseline
play the same starts. How often each type happens comes from a `--weights` file of situation
rates (`goal_ledger --rates`): real Division B matches (default) or our own round-robin.

What the effect is not: the bank's starts are fixed, so a candidate that changes how often
situations happen (wins the ball higher up, concedes fewer free kicks) is credited only for
what it does from each one. A goal counts for every situation that started in the 10 s
before it, so the effect over-counts goals by about 1.2x under sim rates and 1.6x under
real ones (`docs/STRATEGY_DEVELOPMENT.md`). It is a screen; round-robins confirm.

This is the "outer loop, fast half" from `docs/roadmap.md` item 14 — a
benchmark like `tools/motion_planning_benchmark.py`, non-blocking, reporting
numbers rather than gating a merge.

Bank sources, in the order this tool can use them today:
  - Hand-authored anchors (`utama_core.scenario_bench.hand_authored_scenarios`) —
    loaded when no bank is given, no dependency on any replay data; they have no
    situation type or opponent (pass `--opponent`), and count only in the per-start table.
  - Harvested starts (`--harvest-from RUN_DIR`): every restart and ball won of
    a tagged, trustworthy tournament run (`.stats.json` with zero
    `stall_events`, see `utama_core.scenario_bench.scenario_harvester`'s module
    docstring for why this gate exists), up to `--per-situation N` of each type.
    NOT run by this tool — point it at an already-completed `round_robin.py` run
    directory.

For each start, both `--candidate` and `--baseline` play the SAME start from config_a's
side, against the opponent it came from (its source match's config_b), so the bank's
opponents are the round-robin's; `--opponent` plays every start against one config
instead. A start whose opponent does not exist on this branch is skipped and counted.

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

Each start's record keeps the scorer's ordinal outcome (GOAL_AGAINST=-3 ... GOAL_FOR=+3,
see `utama_core.scenario_bench.scenario_scorer`), its signals, fouls and stalls; the
per-start `delta` and `--stop-at-t` are on that ordinal.

`--save-bank PATH` persists the currently-loaded scenario set (hand-authored, or
`--harvest-from`, after `--families` and `--where` filtering) to a single JSON
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
the rest. `--workers N` scores (and harvests) N scenarios at a time.

`--reuse` takes a start's outcome from `replays/match_cache/` instead of playing it
when nothing that start runs has changed since it was stored
(`utama_core.replay.fingerprint.bench_key`: the jittered start, both configs' code, the
shared code and environment, the horizon). A baseline config whose code is unchanged
then costs nothing, and only the candidate plays. `--spot-check F` (default 0.05) also
replays that fraction of the reusable starts and compares them with their records; a
difference means the fingerprint missed a dependency, is printed loudly, evicts the
records this run used, and makes the run exit non-zero.
`--play ID [ID ...]` plays the given starts once each (per `--repeats` seed) for
`--candidate` and shows what happened, instead of scoring a bank:
its outcome and signals, and a timeline of the referee, the ball, the robot of each side
nearest it and every change in the candidate's tactic assignments. Each start is played
against its own opponent unless `--opponent` is given. The replay
(`.npz`) and the candidate's match log (`.intentions.jsonl`) are kept under
`replays/scenario_play/<start id>/`, with the timeline as `.trace.txt` and, with
`--render png|mp4`, a picture of the whole window. Use it to test a change in a
situation the round-robin rarely reaches, or to watch a start a bench run flagged.

Still not built: lifecycle promotion (candidate -> validated -> active). The slow half is
`tools/evaluation/evaluate.py`; the ladder is retired.

Run from the repository root, for example:

    # A candidate against the baseline on the current bank, real Division B weights:
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --candidate build_counter_flow_kernel_strategy --baseline build_split_shape_kernel_strategy --workers 15

    # The same, weighted by our own round-robin's situation rates:
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --candidate counter_flow --baseline split_shape --workers 15 \\
        --weights utama_core/scenario_bench/banks/situation_rates_sim.json

    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --list-scenarios --where third=attacking

    # Harvest and freeze a bank (up to 40 starts per situation type, kickoffs 20):
    pixi run python tools/scenario_bench.py --harvest-from replays/tournament_20261011_230119 \\
        --per-situation 40 --workers 15 --save-bank utama_core/scenario_bench/banks/bank_v8.json --list-scenarios

    # Play a start and watch it (timeline printed, replay and picture kept):
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --play counter_flow_vs_split_shape_t426.3_own_ball_won_attacking --candidate high_press --render png

    # A/B a code change: record at commit A, compare at commit B.
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --candidate build_split_shape_kernel_strategy --workers 15
    pixi run python tools/scenario_bench.py --load-bank utama_core/scenario_bench/banks/bank_v8.json \\
        --candidate build_split_shape_kernel_strategy --workers 15 \\
        --against-results scenario_bench_results/scenario_bench_<commit A>.json
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.replay_trace import trace
from utama_core.analysis.goal_ledger import TYPES, WINDOW_S, per_match
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.global_utils.git_utils import git_revision
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
    SIGNALS,
    ScenarioOutcome,
    _resolve_config_name,
    score_scenario,
)
from utama_core.scenario_bench.start import (
    TAGS,
    BenchScenario,
    drop_near_duplicates,
    jittered,
    load_bank,
    matches_where,
    save_bank,
    start_tags,
)

SCHEMA_VERSION = 1
BANKS = REPO_ROOT / "utama_core" / "scenario_bench" / "banks"
DEFAULT_WEIGHTS = BANKS / "situation_rates_real.json"


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
    elif args.harvest_from is not None:
        scenarios, harvest_report = harvest_run_dir(
            args.harvest_from,
            evaluator_version=git_revision() or "unknown",
            per_situation=args.per_situation,
            workers=args.workers,
        )
    else:
        scenarios = list(all_hand_authored_scenarios())

    if args.families:
        wanted = set(args.families)
        scenarios = [s for s in scenarios if s.provenance.family.value in wanted]
    if args.where:
        scenarios = [s for s in scenarios if matches_where(start_tags(s), args.where)]

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


def start_opponent(bench_scenario: BenchScenario, override: Optional[str] = None) -> Optional[str]:
    """Who a start is played against: `override` (`--opponent`), else the opponent its situation
    came from (its source match's config_b). None when that config does not exist here (a
    start harvested from another branch's strategies) or the start has no source match."""
    name = override or bench_scenario.scenario.config_b_name
    if name is None:
        return None
    try:
        return _resolve_config_name(name)
    except ValueError:
        return None


PLAY_DIR = "scenario_play"  # under REPLAY_BASE_PATH: where --play keeps its replays


def _intention_lines(records: Iterable[dict], t0: float) -> list[tuple[float, str]]:
    """The candidate's tactic assignments from its match log, one line per change, on
    the replay's clock (seconds from its first frame, `t0` in sim time)."""
    out = []
    for r in records:
        if r.get("event") == "intention":
            t = r["sim_time"] - t0
            out.append((t, f"{t:7.2f} tactic {r['tactic_id']} = {r.get('note') or '?'} {r['robot_ids']}"))
    return out


def merge_timeline(trace_lines: list[str], intention_lines: list[tuple[float, str]]) -> list[str]:
    """`replay_trace.trace` lines and `_intention_lines` in time order; at the same time,
    the trace line (the state) comes before the tactic change it led to."""
    rows = [(float(line.split()[0]), 0, line) for line in trace_lines]
    rows += [(t, 1, line) for t, line in intention_lines]
    return [line for _t, _k, line in sorted(rows, key=lambda r: (r[0], r[1]))]


def play_start(
    bench_scenario: BenchScenario,
    *,
    candidate: str,
    opponent: str,
    horizon_s: float,
    seed: int = 0,
    render: Optional[str] = None,
    every: int = 15,
) -> dict:
    """Play one start (`jittered` by `seed`) with a replay and match log recorded, and
    return its result, the timeline and where the files are."""
    tag = f"{_resolve_config_name(candidate)[6:-16]}_vs_{_resolve_config_name(opponent)[6:-16]}_s{seed}"
    name = f"{PLAY_DIR}/{bench_scenario.scenario_id}/{tag}"
    r = score_scenario(
        jittered(bench_scenario, seed),
        candidate_config=candidate,
        opponent_config=opponent,
        horizon_s=horizon_s,
        record=name,
    )
    out = {"scenario_id": bench_scenario.scenario_id, "seed": seed, "result": _result_dict(r), "timeline": []}
    npz = REPLAY_BASE_PATH / f"{name}.npz"
    if r.error is not None or not npz.exists():
        return out
    log = npz.with_suffix(".intentions.jsonl")
    ts = np.load(npz)["ts"]
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    timeline = merge_timeline(trace(npz, 0.0, math.inf, every), _intention_lines(records, float(ts[0])))
    trace_path = npz.with_suffix(".trace.txt")
    trace_path.write_text("\n".join(timeline) + "\n")
    out.update(timeline=timeline, replay=str(npz), log=str(log), trace=str(trace_path))
    # imported here: matplotlib and pygame's drawing aren't needed to score a bank
    from utama_core.analysis.render_clip import render_clip
    from utama_core.analysis.render_window import render_window

    if render == "png":
        out["render"] = render_window(npz, float(ts[0]), float(ts[-1]), npz.with_suffix(".png"))
    elif render == "mp4":
        out["render"] = render_clip(npz, float(ts[0]), float(ts[-1]), npz.with_suffix(".mp4"))
    return out


def _print_play(play: dict, lead_in_s: float) -> None:
    r = play["result"]
    print(f"\n=== {play['scenario_id']} (seed {play['seed']}) ===")
    if r.get("error"):
        print(f"error: {r['error']}")
        return
    print(
        f"outcome {r['outcome']}{' (stalled)' if r.get('stalled') else ''}{' (foul)' if r.get('foul') else ''}; "
        f"scored after a {lead_in_s:g} s lead-in"
    )
    print("signals: " + ", ".join(f"{k} {v:g}" for k, v in (r.get("signals") or {}).items()))
    print("\n".join(play["timeline"]))
    for key in ("replay", "log", "trace", "render"):
        if key in play:
            print(f"{key}: {play[key]}")


def _start_result(bench_scenario: BenchScenario, config: str, opponent: str, horizon_s: float) -> dict:
    return _result_dict(
        score_scenario(bench_scenario, candidate_config=config, opponent_config=opponent, horizon_s=horizon_s)
    )


def _result_dict(r) -> dict:
    return {
        "outcome": int(r.outcome),
        "foul": bool(r.foul),
        "stalled": bool(r.stalled),
        "error": r.error,
        "signals": r.signals,
    }


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
        "signals": [r.get("signals") for r in results],
        "fouls": sum(r["foul"] for r in results),
        "stalls": sum(r["stalled"] for r in results),
        "errors": [r["error"] for r in results if r["error"]],
    }


def _load_against(path: Path, *, opponent: str, horizon_s: float, repeats: int) -> tuple[dict[str, dict], dict]:
    """Baseline outcomes and signals from an earlier run's JSON (typically another commit),
    keyed by scenario id. Refuses a file scored against a different opponent/horizon/repeats:
    those outcomes are not comparable."""
    payload = json.loads(path.read_text())
    for key, want in (("opponent", opponent), ("horizon_s", horizon_s), ("repeats", repeats)):
        if payload.get(key) != want:
            raise SystemExit(f"--against-results {path}: {key} is {payload.get(key)!r}, this run uses {want!r}")
    runs = {
        row["scenario_id"]: {"outcomes": row["candidate_outcomes"], "signals": row.get("candidate_signals")}
        for row in payload["results"]
        if not row.get("candidate_errors")
    }
    source = {"path": str(path), "git_revision": payload.get("git_revision"), "candidate": payload.get("candidate")}
    return runs, source


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
    opponent = start_opponent(bench_scenario, opponent)
    cand = _runs(bench_scenario, candidate, opponent, horizon_s, repeats, plan, fresh)
    base = _runs(bench_scenario, baseline, opponent, horizon_s, repeats, plan, fresh) if baseline else None
    return cand, base, fresh


class _Reuse:
    """`--reuse`'s bookkeeping for one bench run: every start's key, the records it may
    reuse, the ones it replays to check, and what it played."""

    def __init__(self, scenarios, configs, opponent: Optional[str], horizon_s: float, repeats: int, spot_check: float):
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
                start_opponent(bs, opponent),
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
    against: Optional[dict[str, dict]] = None,
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
        base_run = base or (against or {}).get(sid)
        base_outcomes = base_run["outcomes"] if base_run else None
        delta = signal_deltas = None
        # a run that errored (e.g. the start could not be set up) is not an outcome
        if base_outcomes is not None and not cand["errors"] and not (base and base["errors"]):
            delta = statistics.fmean(cand["outcomes"]) - statistics.fmean(base_outcomes)
            signal_deltas = _signal_deltas(cand.get("signals"), base_run.get("signals"))
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
                "situation": bench_scenario.provenance.situation,
                "trigger": bench_scenario.provenance.trigger.value,
                "perspective": bench_scenario.provenance.perspective,
                "candidate_outcomes": cand["outcomes"],
                "baseline_outcomes": base_outcomes,
                "delta": delta,
                "candidate_signals": cand.get("signals"),
                "baseline_signals": base_run.get("signals") if base_run else None,
                "signal_deltas": signal_deltas,
                "candidate_fouls": cand["fouls"],
                "candidate_stalls": cand["stalls"],
                "baseline_fouls": base["fouls"] if base else None,
                "baseline_stalls": base["stalls"] if base else None,
                "candidate_errors": cand["errors"],
                "baseline_errors": base["errors"] if base else [],
            }
        )


def _signal_deltas(cand: Optional[list], base: Optional[list]) -> Optional[dict[str, float]]:
    """Per signal, the candidate's mean over the jittered starts minus the baseline's; None
    when either side has no signals (an older results file)."""
    if not cand or not base or any(s is None for s in cand + base):
        return None
    return {name: statistics.fmean(s[name] for s in cand) - statistics.fmean(s[name] for s in base) for name in SIGNALS}


def _signal_summary(rows: list[dict]) -> dict[str, dict]:
    """Per signal: mean paired delta over the scenarios that have one, its t, and how many
    scenarios it changed. Like the outcome delta, |t| under about 2 is within chance."""
    paired = [r["signal_deltas"] for r in rows if r.get("signal_deltas")]
    summary = {}
    for name in SIGNALS:
        deltas = [d[name] for d in paired]
        summary[name] = {
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "t": _t(deltas) if len(deltas) > 1 else None,
            "changed": sum(1 for d in deltas if d != 0),
            "n_paired": len(deltas),
        }
    return summary


def _type_order(situation: str) -> int:
    return TYPES.index(situation) if situation in TYPES else len(TYPES)


def _goal_rates(outcomes: list[int]) -> tuple[float, float]:
    """(P(for), P(against)) over one start's runs: the first goal within the horizon."""
    return (
        statistics.fmean(o == ScenarioOutcome.GOAL_FOR for o in outcomes),
        statistics.fmean(o == ScenarioOutcome.GOAL_AGAINST for o in outcomes),
    )


def _mean_se(values: list[float]) -> tuple[float, Optional[float]]:
    """Mean and its standard error; no standard error from one value."""
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def situation_effect(rows: list[dict], weights: dict) -> dict:
    """The candidate's effect on goals per match (module docstring), from the paired starts of
    `rows` that have a situation type. Per type: n starts; the candidate's and the baseline's
    P(for) and P(against); the mean paired difference of each, with its standard error across
    the type's starts; how often the type happens per match under `weights`
    (`goal_ledger.situation_rates`); and what it adds to goals scored, conceded and the goal
    difference per match. The total's standard error takes the types as independent, and is
    None when a type has a single start."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        if (
            r.get("situation")
            and r["baseline_outcomes"] is not None
            and not (r["candidate_errors"] or r["baseline_errors"])
        ):
            by_type.setdefault(r["situation"], []).append(r)
    types = {}
    for typ in sorted(by_type, key=_type_order):
        cand = [_goal_rates(r["candidate_outcomes"]) for r in by_type[typ]]
        base = [_goal_rates(r["baseline_outcomes"]) for r in by_type[typ]]
        d_for, se_for = _mean_se([c[0] - b[0] for c, b in zip(cand, base)])
        d_against, se_against = _mean_se([c[1] - b[1] for c, b in zip(cand, base)])
        d_net, se_net = _mean_se([(c[0] - c[1]) - (b[0] - b[1]) for c, b in zip(cand, base)])
        f = per_match(weights, typ)
        types[typ] = {
            "n": len(cand),
            "per_match": f,
            "candidate_p_for": statistics.fmean(c[0] for c in cand),
            "candidate_p_against": statistics.fmean(c[1] for c in cand),
            "baseline_p_for": statistics.fmean(b[0] for b in base),
            "baseline_p_against": statistics.fmean(b[1] for b in base),
            "delta_p_for": d_for,
            "delta_p_for_se": se_for,
            "delta_p_against": d_against,
            "delta_p_against_se": se_against,
            "scored_per_match": f * d_for,
            "conceded_per_match": f * d_against,
            "net_per_match": f * d_net,
            "net_per_match_se": None if se_net is None else f * se_net,
        }
    ses = [t["net_per_match_se"] for t in types.values()]
    total = {
        key: sum(t[key] for t in types.values()) for key in ("scored_per_match", "conceded_per_match", "net_per_match")
    }
    total["net_per_match_se"] = None if None in ses else math.sqrt(sum(se * se for se in ses))
    return {"types": types, "total": total}


def load_weights(path: Path) -> dict:
    """A situation rates file (`goal_ledger --rates`), with the `path` it came from."""
    return {"path": str(path), **json.loads(Path(path).read_text())}


def _effect_line(effect: dict, weights: dict) -> Optional[str]:
    """One line: the effect on goal difference per match, its standard error, and the weights."""
    if not effect["types"]:
        return None
    total = effect["total"]
    se = total["net_per_match_se"]
    return (
        f"Effect on goal difference per match: {total['net_per_match']:+.2f}"
        + (f" ± {se:.2f}" if se is not None else "")
        + f" (scored {total['scored_per_match']:+.2f}, conceded {total['conceded_per_match']:+.2f}); "
        f"weights {Path(weights['path']).name}, a match being {weights['live_min_per_match']:.1f} live minutes"
    )


def _aggregate_by_situation(rows: list[dict]) -> list[dict]:
    """Stalls and signals per situation type (per family for a start with none)."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("situation") or row["family"], []).append(row)
    return [
        {
            "situation": key,
            "n_scenarios": len(grouped[key]),
            "candidate_stalls": sum(r["candidate_stalls"] for r in grouped[key]),
            "signals": _signal_summary(grouped[key]),
        }
        for key in sorted(grouped, key=_type_order)
    ]


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


def _signal_cell(summary: dict) -> str:
    if summary["mean_delta"] is None:
        return "-"
    t = summary["t"]
    if t is None or not math.isfinite(t):
        return _fmt(summary["mean_delta"])
    return f"{_fmt(summary['mean_delta'])} (t {t:+.1f})"


def _signal_report(payload: dict) -> list[str]:
    """The signals table: per signal, the mean paired delta per start and its t, overall and
    per situation."""
    if not payload.get("signals") or not any(s["n_paired"] for s in payload["signals"].values()):
        return []
    groups = [("all", payload["signals"])] + [(a["situation"], a["signals"]) for a in payload["aggregates"]]
    lines = [
        "",
        "## Signals",
        "",
        "Mean paired delta per start (candidate minus baseline) of the candidate's signals over the "
        "scored window, see `docs/signals.md`. More is usually better for "
        + ", ".join(n for n, d in SIGNALS.items() if d > 0)
        + "; less for "
        + ", ".join(n for n, d in SIGNALS.items() if d < 0)
        + ".",
        "",
        "| Situation | " + " | ".join(SIGNALS) + " |",
        "|---|" + "---:|" * len(SIGNALS),
    ]
    for label, signals in groups:
        lines.append(f"| {label} | " + " | ".join(_signal_cell(signals[name]) for name in SIGNALS) + " |")
    return lines


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
        "Opponent: "
        + (f"`{payload['opponent']}`" if payload["opponent"] else "each start's own (its source match's config_b)")
        + "  ",
        f"Horizon: {payload['horizon_s']}s, {payload.get('repeats', 1)} jittered starts per scenario  ",
        f"Weights: `{payload['weights']['path']}` ({payload['weights']['matches']} matches; a match is "
        f"{payload['weights']['live_min_per_match']:.1f} live minutes)  ",
        f"Scenarios scored: {payload.get('n_scored', payload['n_scenarios'])} of {payload['n_scenarios']}"
        + (f" (stopped early: {payload['stopped']})" if payload.get("stopped") else "")
        + (
            f"; {payload['skipped_no_opponent']} more skipped, their opponent not on this branch"
            if payload.get("skipped_no_opponent")
            else ""
        ),
        "",
        "A start's outcome is the first goal within the horizon: for the candidate, against it, or none. "
        "Per situation type, P(for) and P(against) are means over its starts, and the change is paired "
        "(candidate minus baseline on the same starts, ± one standard error across them). Per match, the "
        "change is weighted by how often the type happens in a match under the weights. A goal counts for "
        "every situation in the 10 s before it, so goals per match are over-counted (about 1.2x under sim "
        "weights, 1.6x under real ones). Stalls are counted, not ranked. A screen, not a ranking: "
        "round-robins confirm.",
        "",
        *([f"**{_effect_line(payload['effect'], payload['weights'])}**", ""] if payload["effect"]["types"] else []),
        "## By situation",
        "",
        "| Situation | N | Per match | Candidate P(for) / P(against) | Baseline P(for) / P(against) | "
        "ΔP(for) | ΔP(against) | Scored /match | Conceded /match | Difference /match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for typ, t in payload["effect"]["types"].items():
        lines.append(
            f"| {typ} | {t['n']} | {t['per_match']:.2f} | {t['candidate_p_for']:.2f} / {t['candidate_p_against']:.2f} | "
            f"{t['baseline_p_for']:.2f} / {t['baseline_p_against']:.2f} | "
            f"{_fmt(t['delta_p_for'])} ± {_fmt(t['delta_p_for_se'], '.2f')} | "
            f"{_fmt(t['delta_p_against'])} ± {_fmt(t['delta_p_against_se'], '.2f')} | "
            f"{_fmt(t['scored_per_match'])} | {_fmt(t['conceded_per_match'])} | "
            f"{_fmt(t['net_per_match'])} ± {_fmt(t['net_per_match_se'], '.2f')} |"
        )
    stalls = [f"{a['situation']} {a['candidate_stalls']}" for a in payload["aggregates"] if a["candidate_stalls"]]
    lines += ["", f"Candidate stalls: {', '.join(stalls) if stalls else 'none'}."]

    lines.extend(_signal_report(payload))
    lines.extend(
        [
            "",
            "## Per-scenario",
            "",
            "Outcomes on the scorer's ordinal scale (GOAL_AGAINST=-3 ... NEUTRAL=0 ... GOAL_FOR=+3).",
            "",
            "| Scenario | Situation | Candidate | Baseline | Delta | Stalls |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            f"| {row['scenario_id']} | {row.get('situation') or row['family']} | {row['candidate_outcomes']} | "
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


def _parse_where(parser: argparse.ArgumentParser, terms: Optional[list[str]]) -> dict[str, set[str]]:
    where: dict[str, set[str]] = {}
    for term in terms or []:
        key, sep, values = term.partition("=")
        if not sep or key not in TAGS or not values:
            parser.error(f"--where {term!r}: expected TAG=VALUE[,VALUE] with TAG one of {', '.join(TAGS)}")
        where[key] = set(values.split(","))
    return where


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", help="candidate strategy config, e.g. build_split_shape_kernel_strategy")
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
    parser.add_argument(
        "--opponent",
        help="play every start against this config (default: each start's own opponent, its source match's config_b)",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=WINDOW_S,
        help=f"sim seconds ticked per scenario (default {WINDOW_S:g}, the goal ledger's window)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="situation rates to weight the effect with (goal_ledger --rates; default: real Division B, "
        f"{DEFAULT_WEIGHTS.name}; ours: situation_rates_sim.json)",
    )
    parser.add_argument(
        "--harvest-from",
        type=Path,
        default=None,
        help="a completed round_robin.py run dir to harvest restart scenarios from",
    )
    parser.add_argument(
        "--per-situation",
        type=int,
        default=40,
        help="with --harvest-from, keep up to this many starts of each situation type, kickoffs half as many "
        "(scenario_harvester.sample_by_situation; default 40)",
    )
    parser.add_argument("--families", nargs="+", default=None, help="restrict to these ScenarioFamily values")
    parser.add_argument(
        "--where",
        nargs="+",
        default=None,
        metavar="TAG=VALUE[,VALUE]",
        help="keep only starts whose situation tags match, e.g. third=defensive ball=theirs,loose "
        f"(tags: {', '.join(TAGS)}; see start.start_tags and docs/pitch_zones.md)",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true", help="print the loaded bank with each start's tags, and exit"
    )
    parser.add_argument(
        "--play",
        nargs="+",
        default=None,
        metavar="ID",
        help="play these starts of the loaded bank (one per --repeats seed) for --candidate against its opponent, "
        "print each one's outcome, signals and timeline, and keep its replay under replays/scenario_play/",
    )
    parser.add_argument(
        "--render",
        choices=("png", "mp4"),
        default=None,
        help="with --play, also draw each start's whole window: trails (png) or a clip (mp4, needs ffmpeg)",
    )
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
        if args.candidate is None:
            parser.error("--candidate required unless --list-scenarios")
        if args.baseline and args.against_results:
            parser.error("--baseline and --against-results are alternatives")
    if args.play is not None and (args.baseline or args.against_results or args.stop_at_t is not None):
        parser.error("--play shows starts for --candidate alone; it takes no baseline or --stop-at-t")
    if args.render is not None and args.play is None:
        parser.error("--render needs --play")
    args.where = _parse_where(parser, args.where)
    if args.merge_into is not None and (args.harvest_from is None or args.save_bank is None):
        parser.error("--merge-into needs --harvest-from and --save-bank")

    return args


def main() -> int:
    args = parse_args()
    scenarios, merged, harvest_report = _load_bank(args)
    _save(args, merged + scenarios)

    if args.list_scenarios:
        for bs in scenarios:
            tags = " ".join(f"{k}={v}" for k, v in start_tags(bs).items())
            print(f"{bs.scenario_id:<58} {bs.provenance.situation or bs.provenance.family.value:<36} {tags}")
        print(f"{len(scenarios)} starts")
        if harvest_report:
            print(f"\nHarvest: {harvest_report}")
        return 0

    if not scenarios:
        print("No scenarios loaded (check --families / --harvest-from).", file=sys.stderr)
        return 1

    if args.play is not None:
        by_id = {bs.scenario_id: bs for bs in scenarios}
        unknown = [sid for sid in args.play if sid not in by_id]
        if unknown:
            print(f"Not in the loaded bank: {', '.join(unknown)} (see --list-scenarios)", file=sys.stderr)
            return 1
        failed = False
        for sid in args.play:
            opponent = start_opponent(by_id[sid], args.opponent)
            if opponent is None:
                print(f"{sid}: its opponent is not on this branch; pass --opponent", file=sys.stderr)
                failed = True
                continue
            for seed in range(args.repeats):
                play = play_start(
                    by_id[sid],
                    candidate=args.candidate,
                    opponent=opponent,
                    horizon_s=args.horizon,
                    seed=seed,
                    render=args.render,
                )
                _print_play(play, by_id[sid].lead_in_s)
                failed |= play["result"]["error"] is not None
        return 1 if failed else 0

    generated_at = datetime.now(timezone.utc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")

    n_loaded = len(scenarios)
    scenarios = [bs for bs in scenarios if start_opponent(bs, args.opponent) is not None]
    if len(scenarios) < n_loaded:
        print(f"{n_loaded - len(scenarios)} of {n_loaded} starts skipped: their opponent is not on this branch")
    weights = load_weights(args.weights)

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
    effect = situation_effect(rows, weights)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "git_revision": git_revision(),
        "candidate": args.candidate,
        "baseline": args.baseline,
        "against": against_source,
        "opponent": args.opponent,
        "horizon_s": args.horizon,
        "repeats": args.repeats,
        "weights": weights,
        "n_scenarios": len(scenarios),
        "n_scored": len(rows),
        "skipped_no_opponent": n_loaded - len(scenarios),
        "stop_at_t": args.stop_at_t,
        # why scoring ended early: "detected" (|t| reached stop_at_t) or "futile" (see
        # FUTILE_BELOW); None when every scenario was scored
        "stopped": stopped,
        "reuse": reuse_report,
        "effect": effect,
        "aggregates": _aggregate_by_situation(rows),
        "signals": _signal_summary(rows),
        "results": rows,
        "harvest_report": harvest_report,
    }

    json_path = args.output_dir / f"scenario_bench_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = args.output_dir / f"scenario_bench_{timestamp}.md"
    md_path.write_text(_markdown_report(payload))

    print(f"\nWrote {json_path}\nWrote {md_path}")
    if _effect_line(effect, weights):
        print(_effect_line(effect, weights) + (f" (stopped early: {stopped})" if stopped else ""))
    return 1 if reuse_report and reuse_report["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
