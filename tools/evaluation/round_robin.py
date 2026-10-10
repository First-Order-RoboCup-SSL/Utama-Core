"""round_robin.py (formerly smoke_tournament.py, before that tournament.py) — Round-robin every
`build_*_kernel_strategy` config against every other, one full match per pair
(two halves, see `MAX_MATCH_SECONDS` below; `ladder.py` also sweeps side and kickoff, for one candidate).

Match construction (build strategies, referee, StrategyRunner, kickoff
ceremony, run_dir file layout) lives in `match.py`, shared with `ladder.py`.

Run:
    pixi run python tools/evaluation/round_robin.py [config ...] [flags]

Configs are short names (`tiki_taka`); none means every config not in RETIRED. Flags:
    --pair A B              one match, A as config_a
    --reuse                 take unchanged matches from replays/match_cache/; check the
                            "N reused ... M to play" line it prints before waiting on M
    --spot-check F          with --reuse, replay this fraction of reused matches to verify
    --strict                fail on any stall
    --stop-at-first-stall   stop the run at the first stall
    --both-sides            play each pair twice, sides swapped
    --control-scheme NAME   planner (fpp, trajsample, ...)
    --fuzz-restarts SEED    inject random legal restarts; --fuzz-interval LO HI sets the gap
    --max-workers N, --sequential, --no-save, -v/--verbose, -h/--help

What this does
--------------
Headless rsim, no external process required. For each distinct pair of the
`build_*_kernel_strategy` factories in `utama_core.strategy.kernel_strategy`,
runs one `StrategyRunner` match (6v6: 1 goalkeeper + 5 outfield robots per
side — enough for every factory's minimum, including the `min_attack=2`
configs and `build_three_slot_kernel_strategy`'s three concurrent slots),
steps it until `CustomReferee` calls full time, and records the final
score from its scoreboard.

Deliberately the smallest mechanism that answers "how do these configs do
against each other": a for-loop over `StrategyRunner` matches and a plain
tally, not a new `Tournament`/`Runner` class, no bracketing/seeding. Round-
robin (every pair once) rather than anything more elaborate — there are only
`C(n, 2)` pairs for the current catalog size, all of which comfortably fit
in one run. See `docs/roadmap.md`'s "Multi-strategy / tournament evaluation
infra" section for why this stays this small: the kernel's
`Strategy`/`Tactic`/`Partitioner` primitives already solve the "many things
owning dynamic robot subsets" problem this would otherwise need to
reinvent — a tournament script is purely a driver on top of
`StrategyRunner`'s existing `opp_strategy` support, not a new mechanism.

Every run records the full observability stack (structured intention log,
aggregate stats, replay trail — see `utama_core.engine.match_log`/
`match_stats`, `utama_core.replay`) per match under
`replays/tournament_<UTC-timestamp>/`, plus one `summary.json` for the whole
run — on by default, since it's cheap and is what a tournament run is
usually for. Pass `--no-save` for a quick throwaway run (e.g. a smoke test
of a code change) that doesn't need the trace kept afterward — replays in
particular can be tens to hundreds of MB per match, and a `--no-save` run
that's forgotten about is exactly how `replays/` silently filled up with
unreferenced gigabytes before (cleaned up 2026-08-26; see
`docs/STRATEGY_DEVELOPMENT.md`'s Observability section). `--verbose`/`-v`
also prints a possession/shots/ball-travel line per match as it completes,
independent of `--no-save`.

`--both-sides` plays each pair twice — once with each config as
`config_a` (yellow, defending the right goal in the first half per `run_match`'s
hardcoded `my_team_is_yellow=True, my_team_is_right=True`; the teams change ends at
half-time) — instead of
once. This isn't a repeat: the sim's initial formation and every
`my_team_is_right`-relative geometry call (`enemy_goal_line`, defense-area
clamps, etc.) genuinely differ by side, so re-running the exact same pair
gives a byte-identical result (fixed initial formation, no RNG source that
varies run to run) while swapping which config plays which side gives a
real second data point per pair — the smallest available source of
variance reduction without inventing a noise model. Off by default since it
doubles match count.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import (  # noqa: F401 -- re-exported for callers importing this module
    _CONFIG_NAMES,
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    TICKS_PER_SECOND,
    MatchResult,
    _short_name,
    _stats_to_dict,
)
from tools.evaluation.match import run_match as _lib_run_match
from utama_core.analysis import chances, restart_outcomes, turnover_breakdown
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.replay import match_cache
from utama_core.replay.fingerprint import CodeGraph, match_key

# Left out of a round-robin unless named: the bottom ten of the 2026-10-05 full round-robin
# (docs/strategies.md). Their code stays, since kept strategies and tests build on some of them.
# Named here, outside the match cache key, so retiring one reruns nothing.
RETIRED = {
    "build_tiki_taka_kernel_strategy",
    "build_counter_press_kernel_strategy",
    "build_high_line_zone_kernel_strategy",
    "build_three_slot_kernel_strategy",
    "build_switch_of_play_kernel_strategy",
    "build_low_block_kernel_strategy",
    "build_zone_fluid_kernel_strategy",
    "build_score_aware_zone_flow_kernel_strategy",
    "build_overload_flow_kernel_strategy",
    "build_shadow_switch_kernel_strategy",
}

# A full match, as the rulebook has it: two halves of 300 s of playing time, the clock stopped
# whenever no team may play the ball (`custom_referee/state_machine.py`), so about 710 s of
# sim time (785 s in two measured matches, tournament_20261006_111922). config_a starts on the
# right and kicks off the first half; at half-time the teams change ends and config_b kicks
# off the second. Without `--both-sides` each pair plays once. Shorter matches rank differently: 65s matches were
# 55% draws, and 16 of 40 full matches (replays/tournament_20261003_102921) changed
# result after 180s, enough to reorder the top of the table. rsim is deterministic, so
# a short match is exactly the start of the full one; it just stops before it's decided.
# This only caps a match that can't reach full time (most took under 870 s of sim time).
MAX_MATCH_SECONDS = 1200.0


def run_match(
    config_a_name: str,
    config_b_name: str,
    run_dir: Optional[Path] = None,
    control_scheme: str = "fpp",
    fuzz_seed: Optional[int] = None,
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0),
) -> MatchResult:
    """Play one full match (capped at this module's `MAX_MATCH_SECONDS`), config_a
    fixed to the right side and kickoff (this module's historical, un-swept
    convention — `ladder.py` sweeps side and kickoff). Thin wrapper over `match.run_match`; see its
    docstring for what each parameter does.
    """
    return _lib_run_match(
        config_a_name,
        config_b_name,
        duration_seconds=MAX_MATCH_SECONDS,
        run_dir=run_dir,
        control_scheme=control_scheme,
        fuzz_seed=fuzz_seed,
        fuzz_interval_s=fuzz_interval_s,
    )


def main() -> None:
    # Optional CLI args: config names (with or without the `build_`/
    # `_kernel_strategy` wrapping) to run instead of the full auto-discovered
    # catalog — useful for a quick check of one or two configs without
    # waiting on every pair, e.g. `python tools/evaluation/round_robin.py default
    # low_block`. `--sequential` forces the old one-process-at-a-time loop
    # (useful for debugging a specific match without pool noise); otherwise
    # matches run in a process pool since each `run_match` call is fully
    # self-contained (its own StrategyRunner, its own rSim subprocess, no
    # shared state with any other match) — this is the "many independent
    # matches" case, not the harder "parallelize robots within one match"
    # case (shared per-tick obstacle state, GIL-bound Python, would need its
    # own design). No args keeps today's behaviour: every config, once each.
    # `--max-workers N` caps the pool size, e.g. to leave headroom on a shared
    # machine — os.cpu_count() ignores CPU affinity/cgroup limits, so a
    # `taskset`-restricted run would otherwise still size the pool for all
    # cores and oversubscribe.
    # `--pair A B` plays just that one fixture, A as config_a.
    # `--both-sides` plays every pair twice, once with each config on each
    # side — see the module docstring for why this is real variance
    # reduction rather than a duplicate match.
    # `--control-scheme NAME` runs every match with that motion-control scheme
    # (e.g. `fpp`, `dwa`, `trajsample` — see
    # utama_core.motion_planning.src.common.control_schemes) on both sides
    # instead of StrategyRunner's default `fpp`. This script compares
    # strategies against each other, not planners against each other, so
    # there's a single scheme per run rather than a per-side override.
    # `--verbose`/`-v` prints each match's possession/shots/ball-travel line
    # alongside the score. The full observability stack (intention log, full
    # stats JSON, replay trail — see `utama_core.engine.match_log`/
    # `match_stats`, `utama_core.replay`) is recorded by default, under
    # `replays/tournament_<timestamp>/`: it's cheap per match and is the
    # point of running a tournament at all — a bare win/loss tally without
    # the trace behind it was exactly what made the earlier scoreless-match
    # investigations start from zero every time. `--no-save` skips all of
    # it (no run_dir created, no summary.json) for a throwaway smoke-test
    # run that doesn't need to be analyzed afterward.
    # `--fuzz-restarts SEED` swaps in `RestartFuzzingReferee` for every match
    # in this run (see docs/custom_referee.md's "Restart fuzzing" section /
    # `utama_core/custom_referee/restart_fuzzer.py`), injecting extra
    # seeded-random legal restarts during live play so referee auto-advance
    # paths get exercised far more often than natural play alone triggers
    # them. `--fuzz-interval LO HI` sets the sim-second gap range between
    # injections (default 25-45s); both are recorded in summary.json
    # (`fuzz_seed`/`fuzz_interval_s`, null when off) so a fuzzed run is
    # reproducible from the summary alone.
    # `--stop-at-first-stall` exits as soon as any match records a
    # `StallEvent`, instead of finishing all `C(n, 2)` pairs — for iterating
    # on a stall fix without waiting ~40 minutes for the full round-robin;
    # rsim is deterministic, so the first stall found for a given
    # catalog/seed/control-scheme reproduces the same way every run. Implies
    # `--strict`. In pool mode, in-flight matches still finish (their cost is
    # already sunk) but no further matches are submitted.
    # `--reuse` takes a match's result from `replays/match_cache/` instead of
    # playing it when nothing that match runs has changed since it was stored
    # (`utama_core.replay.fingerprint.match_key`; rsim is deterministic, so the
    # stored result is what the match would play). After a change to one
    # strategy only its matches play. `--spot-check F` (default 0.05) also
    # replays that fraction of the reusable matches, at least one, and compares
    # them with their stored records: a difference means the fingerprint missed
    # a dependency, is printed loudly, evicts those records, and fails `--strict`.
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(__doc__)
        return
    sequential = "--sequential" in args
    args = [a for a in args if a != "--sequential"]
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]
    both_sides = "--both-sides" in args
    args = [a for a in args if a != "--both-sides"]
    max_workers_override: int | None = None
    if "--max-workers" in args:
        idx = args.index("--max-workers")
        max_workers_override = int(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]
    no_save = "--no-save" in args
    args = [a for a in args if a != "--no-save"]
    reuse = "--reuse" in args
    args = [a for a in args if a != "--reuse"]
    spot_check = 0.05
    if "--spot-check" in args:
        idx = args.index("--spot-check")
        spot_check = float(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]
    if reuse and no_save:
        # A stored record is a recorded match's (the match log and stats recorder
        # live), and the post-run analyses it carries are read from saved replays.
        raise SystemExit("--reuse stores and reads recorded matches; drop --no-save")
    # `--strict` exits non-zero when any match in this run recorded a stall
    # event (see the STALLS section below) — for CI/pre-merge gating, so a
    # stall regression fails the run instead of only showing up if someone
    # reads the printed section or summary.json by hand.
    strict = "--strict" in args
    args = [a for a in args if a != "--strict"]
    # `--stop-at-first-stall` exits the round-robin as soon as any match
    # records a `StallEvent`, instead of running all `C(n, 2)` pairs — for
    # iterating on a stall fix, where the other ~230 matches add nothing
    # once one reproduction is in hand (rsim is deterministic, so the first
    # stall found for a given catalog/seed/control-scheme is stable across
    # runs). Implies `--strict`, since the point of stopping early is to
    # fail fast, not to keep going and report success. Standings/STALLS
    # output below still reflects however many matches actually ran.
    stop_at_first_stall = "--stop-at-first-stall" in args
    args = [a for a in args if a != "--stop-at-first-stall"]
    strict = strict or stop_at_first_stall
    if stop_at_first_stall and no_save:
        # Stall detection reads `MatchResult.stats`, which is only populated
        # when `run_match` gets a `stats_path` — i.e. never under `--no-save`
        # (see `run_match`'s `extra_kwargs` construction). Combined with
        # `--no-save`, `_stalled()` would be unconditionally False for every
        # match, so this run-forever's fast-exit would silently never fire —
        # worse than a no-op, since it would look like a clean run. `--strict`
        # alone has this same gap but degrades to "did nothing", which is
        # already documented above; `--stop-at-first-stall`'s entire point is
        # to fire, so make the misuse loud instead.
        raise SystemExit("--stop-at-first-stall requires stats collection; drop --no-save")
    control_scheme = "fpp"
    if "--control-scheme" in args:
        idx = args.index("--control-scheme")
        control_scheme = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]
    # `--fuzz-restarts SEED` runs every match with `RestartFuzzingReferee`
    # instead of plain `CustomReferee`, injecting extra seeded-random legal
    # restarts during live play (see docs/custom_referee.md's "Restart
    # fuzzing" section / restart_fuzzer.py). Same seed -> identical injection
    # schedule (kind, team, sim time), so a run is reproducible.
    fuzz_seed: Optional[int] = None
    if "--fuzz-restarts" in args:
        idx = args.index("--fuzz-restarts")
        fuzz_seed = int(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]
    # `--fuzz-interval LO HI` sets the (sim-second) gap range between
    # injections when `--fuzz-restarts` is on. Default 25-45s: about 17
    # injections over a full match, not the 8-20s range in
    # RestartFuzzingReferee's own docstring, which is a stress-test example,
    # not a sane default for a normal round-robin.
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0)
    if "--fuzz-interval" in args:
        idx = args.index("--fuzz-interval")
        fuzz_interval_s = (float(args[idx + 1]), float(args[idx + 2]))
        args = args[:idx] + args[idx + 3 :]

    # `--pair A B` plays exactly one fixture, A as config_a (yellow, right side
    # and kickoff in the first half) -- e.g. to rerun one stalled match from a round-robin, which
    # reproduces exactly since rsim is deterministic.
    pair: Optional[tuple[str, str]] = None
    if "--pair" in args:
        idx = args.index("--pair")
        pair = (args[idx + 1], args[idx + 2])
        args = args[:idx] + args[idx + 3 :]
        if args:
            raise SystemExit(f"--pair takes no other config names: {args}")

    if pair is not None:
        config_names = _resolve_config_names(list(pair))
        pairs = [tuple(_resolve_config_names([name])[0] for name in pair)]
    else:
        if args:
            config_names = _resolve_config_names(args)
            if len(config_names) < 2:
                raise SystemExit("Need at least 2 configs to play a round-robin.")
        else:
            config_names = [name for name in _CONFIG_NAMES if name not in RETIRED]

        base_pairs = list(itertools.combinations(sorted(config_names), 2))
        # --both-sides plays (a, b) and (b, a) as distinct fixtures — see the
        # module docstring for why this is a real second data point (side-
        # dependent geometry) rather than a duplicate, unlike naively re-running
        # the same pair (which is byte-identical: fixed initial formation, no
        # varying RNG source).
        pairs = [(a, b) for a, b in base_pairs] + ([(b, a) for a, b in base_pairs] if both_sides else [])

    run_id = datetime.now(timezone.utc).strftime("tournament_%Y%m%d_%H%M%S")
    run_dir: Optional[Path] = None
    if not no_save:
        run_dir = REPLAY_BASE_PATH / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = pairs

    def _keys() -> dict[tuple[str, str], str]:
        graph = CodeGraph()
        return {
            (a, b): match_key(
                graph,
                a,
                b,
                duration_seconds=MAX_MATCH_SECONDS,
                control_scheme=control_scheme,
                fuzz_seed=fuzz_seed,
                fuzz_interval_s=fuzz_interval_s,
            )
            for a, b in all_pairs
        }

    cache = match_cache.MatchCache()
    keys: dict[tuple[str, str], str] = _keys() if reuse else {}
    stored = {p: rec for p in pairs if reuse and (rec := cache.get(keys[p])) is not None}
    spot_checked = {
        p for p in stored if keys[p] in match_cache.spot_check_sample([keys[q] for q in stored], spot_check)
    }
    reused = {p: rec for p, rec in stored.items() if p not in spot_checked}
    pairs = [p for p in pairs if p not in reused]

    print(
        f"Round-robin: {len(config_names)} configs, {len(all_pairs)} matches" + (" (both sides)" if both_sides else "")
    )
    if reuse:
        print(
            f"--reuse: {len(reused)} stored result(s) reused, {len(spot_checked)} replayed as a spot-check, "
            f"{len(pairs) - len(spot_checked)} to play"
        )
    print(f"{N_OUTFIELD + 1}v{N_OUTFIELD + 1}, two 300 s halves of playing time per match, headless rsim")
    if no_save:
        print("--no-save: not recording replay/intention-log/stats for this run")
    else:
        print(f"Recording to replays/{run_id}/ (per-match replay, intention log, stats)")
    if not sequential:
        default_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = max(1, min(len(pairs), max_workers_override or default_workers))
        print(f"Running {n_workers} matches concurrently (--sequential to disable)\n")
    else:
        print("Running sequentially\n")

    results: list[MatchResult] = []
    wins: dict[str, int] = {name: 0 for name in config_names}
    draws: dict[str, int] = {name: 0 for name in config_names}

    def _stats_line(result: MatchResult) -> str:
        if not result.stats:
            return ""
        s = result.stats
        poss = s["possession_pct"]
        shots = s["shots"]
        return (
            f"      possession {poss['friendly']:.0%}/{poss['enemy']:.0%}  "
            f"shots {shots['friendly']}-{shots['enemy']}  "
            f"ball_travel {s['ball_travel_m']:.1f}m  "
            f"turnovers {s['turnovers']}  completed_passes {s['completed_passes']}  "
            f"attacking_third_entries {s['attacking_third_entries']}"
        )

    def _stalled(result: MatchResult) -> bool:
        return bool(result.stats and result.stats.get("stall_events"))

    def _record(result: MatchResult) -> None:
        results.append(result)
        if result.winner == "draw":
            draws[result.config_a] += 1
            draws[result.config_b] += 1
        else:
            wins[result.winner] += 1
        print(
            f"{result.config_a:<40} {result.score_a} - {result.score_b} "
            f"{result.config_b:<40} winner={result.winner}",
            flush=True,
        )
        if verbose:
            line = _stats_line(result)
            if line:
                print(line, flush=True)

    # Each played match's restart and ball-loss analyses, made as it finishes: the end of the run
    # reads them from here rather than replaying every saved match again.
    analysed: dict[str, tuple[list[dict], dict]] = {}

    def _analyse(result: MatchResult) -> None:
        # Stored in the match cache straight away too, not only once the whole run ends, so a crash
        # (WSL running out of memory killed two runs on 2026-10-09) loses only the matches still
        # playing. A spot-checked match is left to `_store_played`, which compares it with its
        # stored record; so is any match whose key changed since the run started (code edited mid-run).
        pair = (result.config_a, result.config_b)
        npz = run_dir / f"{_tag(pair)}.npz" if run_dir is not None else None
        if npz is None or not npz.exists():  # no run dir, or crashed without a replay
            return
        restarts = restart_outcomes.analyse_match(npz)
        losses = turnover_breakdown.analyse_match(str(npz))
        analysed[_tag(pair)] = (restarts, losses)
        if reuse and pair not in spot_checked and _keys()[pair] == keys[pair]:
            cache.put(keys[pair], {"result": dataclasses.asdict(result), "restarts": restarts, "losses": losses})

    for (a, b), rec in reused.items():
        _record(MatchResult(**rec["result"]))

    if sequential:
        for config_a_name, config_b_name in pairs:
            result = run_match(
                config_a_name,
                config_b_name,
                run_dir=run_dir,
                control_scheme=control_scheme,
                fuzz_seed=fuzz_seed,
                fuzz_interval_s=fuzz_interval_s,
            )
            _record(result)
            _analyse(result)
            if stop_at_first_stall and _stalled(result):
                tag = f"{_short_name(result.config_a)}_vs_{_short_name(result.config_b)}"
                print(f"\n--stop-at-first-stall: stopping after {tag}", flush=True)
                break
    else:
        # Matches complete out of submission order under a process pool —
        # printed as they finish rather than buffered back into pair order,
        # since waiting to preserve order would throttle everything to the
        # slowest in-flight match. Final standings are still sorted, so the
        # only user-visible reordering is the interleaved progress log.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(run_match, a, b, run_dir, control_scheme, fuzz_seed, fuzz_interval_s) for a, b in pairs
            ]
            for future in as_completed(futures):
                result = future.result()
                _record(result)
                _analyse(result)
                if stop_at_first_stall and _stalled(result):
                    tag = f"{_short_name(result.config_a)}_vs_{_short_name(result.config_b)}"
                    print(
                        f"\n--stop-at-first-stall: stopping after {tag} "
                        "(in-flight matches still finish; not resubmitted)",
                        flush=True,
                    )
                    for fut in futures:
                        fut.cancel()
                    break

    print("\nStandings (wins, draws):")
    for name in sorted(config_names, key=lambda n: (-wins[n], -draws[n])):
        print(f"  {name:<40} {wins[name]}W {draws[name]}D")

    # STALLS: every match that recorded a `StallEvent` (see
    # `utama_core.engine.match_stats`'s RESTART_STALL/COMMITTED_FROZEN
    # watchdogs), plus a heuristic backstop for anything the watchdog itself
    # missed — possession pinned at 100%/0% with almost no ball movement is
    # exactly the signature a stalled match showed before this watchdog
    # existed (see this module's/`match_stats.py`'s docs). Both are printed
    # together since they answer the same question ("did this run stall
    # anywhere") and both go into summary.json per match either way.
    def _match_tag_of(r: MatchResult) -> str:
        return f"{_short_name(r.config_a)}_vs_{_short_name(r.config_b)}"

    stalled_matches: list[tuple[MatchResult, list[dict]]] = []
    backstop_matches: list[MatchResult] = []
    for r in results:
        if not r.stats:
            continue
        events = r.stats.get("stall_events") or []
        if events:
            stalled_matches.append((r, events))
        poss = r.stats.get("possession_pct") or {}
        pinned = (poss.get("friendly") == 1.0 and poss.get("enemy") == 0.0) or (
            poss.get("friendly") == 0.0 and poss.get("enemy") == 1.0
        )
        if pinned and (r.stats.get("ball_travel_m") or 0.0) < 1.0:
            backstop_matches.append(r)

    stalled_match_ids = {id(r) for r, _ in stalled_matches}
    incidents = stall_incidents([{"config_a": r.config_a, "config_b": r.config_b, "stats": r.stats} for r in results])
    if stalled_matches or backstop_matches:
        print(f"\nSTALLS: {len(stalled_matches)} match(es), {len(incidents)} distinct incident(s)")
        for r, events in stalled_matches:
            for e in events:
                tactic_str = f" tactics={e['tactic_ids']}" if e["tactic_ids"] else ""
                diagnosis = f" -- {e['diagnosis']}" if e.get("diagnosis") else ""
                print(
                    f"  {_match_tag_of(r):<50} {e['kind']:<17} onset t={e['sim_time']:.1f}s "
                    f"referee={e['referee_command']}{tactic_str}{diagnosis}"
                )
        for r in backstop_matches:
            if id(r) in stalled_match_ids:
                continue
            poss = r.stats["possession_pct"]
            print(
                f"  {_match_tag_of(r):<50} {'POSSESSION_BACKSTOP':<17} "
                f"possession {poss['friendly']:.0%}/{poss['enemy']:.0%} "
                f"ball_travel {r.stats['ball_travel_m']:.2f}m"
            )
    else:
        print("\nSTALLS: none")

    backstop_match_ids = {id(r) for r in backstop_matches}
    summary = {
        "run_id": run_id,
        "run": _run_metadata(),
        "config_names": sorted(config_names),
        "control_scheme": control_scheme,
        "max_match_seconds": MAX_MATCH_SECONDS,
        "fuzz_seed": fuzz_seed,
        "fuzz_interval_s": list(fuzz_interval_s) if fuzz_seed is not None else None,
        "results": [
            {
                "config_a": r.config_a,
                "config_b": r.config_b,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "winner": r.winner,
                "stats": r.stats,
                "possession_backstop": id(r) in backstop_match_ids,
                "reused": (r.config_a, r.config_b) in reused,
            }
            for r in results
        ],
        "standings": {name: {"wins": wins[name], "draws": draws[name]} for name in config_names},
        "stalled_match_count": len(stalled_matches),
        "stall_incidents": incidents,
        "possession_backstop_match_count": len(backstop_matches),
        "reuse": (
            {"reused": len(reused), "spot_checked": len(spot_checked), "played": len(pairs) - len(spot_checked)}
            if reuse
            else None
        ),
    }
    real_losses_by_match: Optional[dict[str, dict[str, int]]] = None
    chances_by_match: dict[str, dict] = {}
    mismatches: list[tuple[str, str]] = []
    if run_dir is not None:
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        # RESTARTS: what became of every restart -- taken, or voided/stopped/timed out.
        restarts_by_match = {tag: restarts for tag, (restarts, _) in analysed.items()}
        restarts_by_match.update({_tag(p): rec["restarts"] for p, rec in reused.items()})
        summary["restarts"] = restart_outcomes.summarise(
            [e for m in sorted(restarts_by_match) for e in restarts_by_match[m]]
        )
        _print_restarts(summary["restarts"])
        # BALL LOSSES: how the friendly side (config_a) gave the ball away, by kind, foul rule
        # and tactic — `MatchStats.turnovers` alone is mostly two robots on one ball flipping
        # "nearest robot". Analysed as each match finished (`_analyse`).
        losses = [loss for _, loss in analysed.values()]
        losses = sorted(losses + [rec["losses"] for rec in reused.values()], key=lambda r: r["match"])
        if reuse:
            mismatches = _store_played(
                cache, keys, _keys(), results, reused, spot_checked, stored, restarts_by_match, losses
            )
            # Kept in the summary, not only printed: a spot-check is also the run's check that
            # rsim replays a match exactly (docs/roadmap.md 10a).
            summary["reuse"]["mismatches"] = sorted(_tag(p) for p in mismatches)
        if losses:  # no replays saved (e.g. every match crashed): nothing to break down
            summary["ball_losses"] = turnover_breakdown.breakdown(losses)
            (run_dir / "ball_losses.md").write_text(turnover_breakdown.report(run_id, losses, summary))
            _print_ball_losses(summary["ball_losses"], run_id)
            real_losses_by_match = {r["match"]: turnover_breakdown.real_loss_kinds(r) for r in losses}
            # Records stored before chances were measured have none: their matches are left out.
            chances_by_match = {r["match"]: r["chances"] for r in losses if "chances" in r}

    summary["strategies"] = strategy_table(summary["results"], real_losses_by_match, chances_by_match)
    _print_strategy_table(summary["strategies"])
    _print_loss_kinds(summary["strategies"])
    _print_chances(summary["strategies"])
    summary["fouls"] = foul_table(summary["results"])
    _print_foul_table(summary["fouls"])
    if run_dir is not None:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results + stats: replays/{run_id}/summary.json")

    if strict and reuse and mismatches:
        raise SystemExit(f"--strict: {len(mismatches)} spot-checked match(es) differ from their stored records")
    if strict and (stalled_matches or backstop_matches):
        raise SystemExit(
            f"--strict: {len(stalled_matches)} match(es) with stall events, "
            f"{len(backstop_matches)} match(es) flagged by the possession backstop"
        )


def _tag(pair: tuple[str, str]) -> str:
    """A match's file stem in a run directory (`match.run_match`'s `match_tag`)."""
    return f"{_short_name(pair[0])}_vs_{_short_name(pair[1])}"


def _store_played(
    cache: match_cache.MatchCache,
    keys: dict[tuple[str, str], str],
    keys_now: dict[tuple[str, str], str],
    results: list[MatchResult],
    reused: dict,
    spot_checked: set,
    stored: dict,
    restarts_by_match: dict[str, list[dict]],
    losses: list[dict],
) -> list[tuple[str, str]]:
    """Store every match this run played, and compare each spot-checked one with its stored
    record. Returns the spot-checked pairs that differ; their records are evicted, not
    replaced, since their key no longer names one result, and so is every record this run
    reused. Stores nothing if any key changed while the run played (code edited mid-run)."""
    if keys_now != keys:
        print("\n--reuse: the code changed while this run played; storing nothing", flush=True)
        return []
    losses_by_match = {r["match"]: r for r in losses}
    mismatches: list[tuple[str, str]] = []
    for r in results:
        pair = (r.config_a, r.config_b)
        if pair in reused or _tag(pair) not in losses_by_match:  # reused, or crashed without a replay
            continue
        record = {
            "result": dataclasses.asdict(r),
            "restarts": restarts_by_match.get(_tag(pair), []),
            "losses": losses_by_match[_tag(pair)],
        }
        if pair in spot_checked:
            parts = match_cache.differences(stored[pair], record)
            if parts:
                mismatches.append(pair)
                cache.evict(keys[pair])
                print(
                    f"\n*** --reuse SPOT-CHECK MISMATCH: {_tag(pair)} differs from its stored record in {parts}. "
                    "The fingerprint missed something this match depends on; its record is evicted. ***",
                    flush=True,
                )
            continue
        cache.put(keys[pair], record)
    if spot_checked:
        print(f"\n--reuse: spot-check {len(spot_checked) - len(mismatches)}/{len(spot_checked)} identical", flush=True)
    if mismatches:
        # The records this run reused were trusted on the same fingerprints that just failed.
        for pair in reused:
            cache.evict(keys[pair])
        print(
            f"*** --reuse: this run's {len(reused)} reused result(s) are evicted too and its standings can't be "
            "trusted; rerun without --reuse. ***",
            flush=True,
        )
    return mismatches


def _resolve_config_names(requested_names: list[str]) -> list[str]:
    """Factory names for `requested_names`, each given with or without the
    `build_`/`_kernel_strategy` wrapping, in catalog order."""
    requested = set(requested_names)
    config_names = [
        name
        for name in _CONFIG_NAMES
        if name in requested or name.removeprefix("build_").removesuffix("_kernel_strategy") in requested
    ]
    unmatched = requested - {
        n for name in config_names for n in (name, name.removeprefix("build_").removesuffix("_kernel_strategy"))
    }
    if unmatched:
        raise SystemExit(f"Unknown config name(s): {sorted(unmatched)}. Available: {sorted(_CONFIG_NAMES)}")
    return config_names


def _run_metadata() -> dict:
    """What produced this run: enough to reproduce it or line it up against another."""

    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv[1:],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def strategy_table(
    results: list[dict],
    real_losses_by_match: Optional[dict[str, dict[str, int]]] = None,
    chances_by_match: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """Per-strategy totals over `summary.json`-shaped `results`. Match stats are from
    config_a's side, so config_b reads the `enemy_*` counterparts. Real ball losses are only
    measured for config_a (the side with an intentions log), so they are totalled over the
    matches a strategy played as config_a (`matches_as_a`), in total and by kind
    (`turnover_breakdown.TURNOVER_KINDS` and `RESTART_KINDS`). `stalled` counts matches that
    recorded a stall event or tripped the possession backstop: their result is still in
    W-D-L, flagged rather than dropped, since a stall can be the strategy's own fault.
    `chances` (`analysis.chances.rates`) is measured for both sides, over the matches in
    `chances_by_match`; `pass_progress_m` is the mean metres gained toward goal per
    completed pass, `forward_pass_share` the share gaining at least 1 m."""
    table: dict[str, dict] = {}
    chance_totals: dict[str, dict[str, float]] = {}
    progress: dict[str, list[float]] = {}
    for r in results:
        stats = r.get("stats") or {}
        fouls = stats.get("fouls_by_side") or {}
        tag = f"{_short_name(r['config_a'])}_vs_{_short_name(r['config_b'])}"
        for name, own, is_a in ((r["config_a"], "friendly", True), (r["config_b"], "enemy", False)):
            row = table.setdefault(
                name,
                {
                    "matches": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "shots": 0,
                    "completed_passes": 0,
                    "attacking_third_entries": 0,
                    "fouls": 0,
                    "stalled": 0,
                    "matches_as_a": 0,
                    "real_losses_as_a": 0,
                    "real_loss_kinds_as_a": {},
                },
            )
            prefix = "" if is_a else "enemy_"
            gf, ga = (r["score_a"], r["score_b"]) if is_a else (r["score_b"], r["score_a"])
            row["matches"] += 1
            row["wins" if gf > ga else "draws" if gf == ga else "losses"] += 1
            row["goals_for"] += gf
            row["goals_against"] += ga
            row["shots"] += (stats.get("shots") or {}).get(own, 0)
            row["completed_passes"] += stats.get(f"{prefix}completed_passes", 0)
            row["attacking_third_entries"] += stats.get(f"{prefix}attacking_third_entries", 0)
            row["fouls"] += sum((fouls.get(own) or {}).values())
            row["stalled"] += bool(stats.get("stall_events") or r.get("possession_backstop"))
            progress.setdefault(name, []).extend(stats.get(f"{prefix}pass_progress_m") or [])
            if chances_by_match and tag in chances_by_match:
                chances.add(chance_totals.setdefault(name, {}), chances.side_totals(chances_by_match[tag], own))
            if is_a and real_losses_by_match is not None and tag in real_losses_by_match:
                row["matches_as_a"] += 1
                kinds = real_losses_by_match[tag]
                row["real_losses_as_a"] += sum(kinds.values())
                for kind, n in kinds.items():
                    row["real_loss_kinds_as_a"][kind] = row["real_loss_kinds_as_a"].get(kind, 0) + n
    for name, row in table.items():
        gains = progress.get(name, [])
        row["pass_progress_m"] = round(sum(gains) / len(gains), 2) if gains else None
        row["forward_pass_share"] = round(sum(g >= 1.0 for g in gains) / len(gains), 2) if gains else None
        row["chances"] = chances.rates(chance_totals.get(name, {}))
    return table


def stall_incidents(results: list[dict]) -> list[dict]:
    """Stall events over `summary.json`-shaped `results`, with deterministic duplicates
    merged: the same kind, onset tick and duration tick in matches that share a strategy
    is one freeze seen against two opponents that hadn't diverged yet (rsim is
    deterministic), not two separate problems. In order of first appearance."""
    incidents: list[dict] = []
    by_key: dict[tuple, dict] = {}
    for r in results:
        tag = f"{_short_name(r['config_a'])}_vs_{_short_name(r['config_b'])}"
        for e in (r.get("stats") or {}).get("stall_events") or []:
            ticks = (e["kind"], round(e["sim_time"] * TICKS_PER_SECOND), round(e["duration_s"] * TICKS_PER_SECOND))
            keys = [(name, *ticks) for name in (r["config_a"], r["config_b"])]
            incident = next((by_key[k] for k in keys if k in by_key), None)
            if incident is None:
                incident = {"kind": e["kind"], "sim_time": e["sim_time"], "duration_s": e["duration_s"], "matches": []}
                incidents.append(incident)
            if tag not in incident["matches"]:
                incident["matches"].append(tag)
            for k in keys:
                by_key.setdefault(k, incident)
    return incidents


def _print_restarts(restarts: dict) -> None:
    if not restarts["restarts"]:
        return
    print(
        f"\nRESTARTS: {restarts['restarts']}, {restarts['reached_normal_start'] / restarts['restarts']:.0%} "
        "reached NORMAL_START"
    )
    for kind, row in restarts["by_kind"].items():
        outcomes = ", ".join(f"{k} {n}" for k, n in row["outcomes"].items())
        print(f"  {kind:<16} {row['n']:>4}  reached {row['reached_normal_start'] / row['n']:>4.0%}  {outcomes}")


def foul_table(results: list[dict]) -> dict[str, dict]:
    """Every logged foul (`MatchStats.fouls`, both sides) by rule, then by
    "strategy/tactic" of the offending robot. `inferred` counts fouls whose rule only
    named a team, attributed to that side's robot nearest the ball."""
    table: dict[str, dict] = {}
    for r in results:
        for foul in (r.get("stats") or {}).get("fouls") or []:
            strategy = _short_name(r["config_a"] if foul["side"] == "friendly" else r["config_b"])
            row = table.setdefault(foul["rule"], {"total": 0, "inferred": 0, "by_tactic": {}})
            row["total"] += 1
            row["inferred"] += bool(foul["inferred"])
            key = f"{strategy}/{foul['tactic'] or 'unknown'}"
            row["by_tactic"][key] = row["by_tactic"].get(key, 0) + 1
    for row in table.values():
        row["by_tactic"] = dict(sorted(row["by_tactic"].items(), key=lambda kv: -kv[1]))
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["total"]))


def _print_foul_table(table: dict[str, dict]) -> None:
    if not table:
        return
    print("\nFOULS (both sides; top strategy/tactic of the offending robot; * = rule named only a team):")
    for rule, row in table.items():
        top = ", ".join(f"{k} {c}" for k, c in list(row["by_tactic"].items())[:4])
        star = "*" if row["inferred"] else ""
        print(f"  {rule + star:<30} {row['total']:>5}  {top}")


def _print_strategy_table(table: dict[str, dict]) -> None:
    print("\nSTRATEGIES (per match; losses = real ball losses, as config_a only):")
    print(
        f"  {'strategy':<32} {'W-D-L':>9} {'GF':>5} {'GA':>5} {'shots':>6} {'passes':>7} {'entries':>8} {'fouls':>6} {'losses':>7} {'stalled':>8}"
    )
    for name, t in sorted(table.items(), key=lambda kv: (-kv[1]["wins"], -kv[1]["draws"])):
        n = max(1, t["matches"])
        losses = f"{t['real_losses_as_a'] / t['matches_as_a']:.1f}" if t["matches_as_a"] else "-"
        print(
            f"  {_short_name(name):<32} {t['wins']:>3}-{t['draws']}-{t['losses']:<3} {t['goals_for'] / n:>5.2f} "
            f"{t['goals_against'] / n:>5.2f} {t['shots'] / n:>6.2f} {t['completed_passes'] / n:>7.1f} "
            f"{t['attacking_third_entries'] / n:>8.1f} {t['fouls'] / n:>6.1f} {losses:>7} {t['stalled']:>8}"
        )


def _print_loss_kinds(table: dict[str, dict]) -> None:
    """Real ball losses per match as config_a, by kind: where each strategy gives the ball away."""
    rows = {name: t for name, t in table.items() if t["matches_as_a"]}
    if not rows:
        return
    totals: dict[str, int] = {}
    for t in rows.values():
        for kind, n in t["real_loss_kinds_as_a"].items():
            totals[kind] = totals.get(kind, 0) + n
    kinds = sorted(totals, key=lambda k: -totals[k])
    short = {
        "shot_saved_or_blocked": "shot_saved",
        "ball_out_after_kick": "out_kick",
        "ball_out_other": "out_other",
        "pass_intercepted": "intercepted",
        "loose_ball_lost": "loose_lost",
    }
    print("\nLOSS KINDS (real ball losses per match as config_a, by kind; see ball_losses.md):")
    print(f"  {'strategy':<32} " + " ".join(f"{short.get(k, k)[:11]:>11}" for k in kinds))
    for name, t in sorted(rows.items(), key=lambda kv: -kv[1]["real_losses_as_a"] / kv[1]["matches_as_a"]):
        n = t["matches_as_a"]
        print(
            f"  {_short_name(name):<32} " + " ".join(f"{t['real_loss_kinds_as_a'].get(k, 0) / n:>11.2f}" for k in kinds)
        )


def _print_chances(table: dict[str, dict]) -> None:
    """Chances created and conceded per strategy, both sides (`analysis.chances`)."""
    rows = {name: t for name, t in table.items() if t["chances"]["matches"]}
    if not rows:
        return

    def pct(v: Optional[float]) -> str:
        return f"{v:.0%}" if v is not None else "-"

    def num(v: Optional[float]) -> str:
        return f"{v:.1f}" if v is not None else "-"

    print(
        "\nCHANCES (both sides; conv = goals per shot, open = goal mouth unblocked at the shot, "
        "regain>shot = open-play regains shot from within 10 s, danger = s/match the enemy held "
        "the ball in our defensive third, fk>shot = free kicks shot from within 10 s, "
        "progress = m gained per completed pass):"
    )
    print(
        f"  {'strategy':<28} {'shots':>6} {'conv':>5} {'dist':>5} {'open':>5} {'save':>5} {'open vs':>7} "
        f"{'regain>shot':>11} {'secs':>5} {'danger':>6} {'fk>shot':>7} {'progress':>8} {'fwd':>4}"
    )
    for name, t in sorted(rows.items(), key=lambda kv: (-kv[1]["wins"], -kv[1]["draws"])):
        c = t["chances"]
        print(
            f"  {_short_name(name):<28} {c['shots'] / c['matches']:>6.1f} {pct(c['conversion']):>5} "
            f"{num(c['shot_distance_m']):>5} {pct(c['shot_open_goal']):>5} {pct(c['save_rate']):>5} "
            f"{pct(c['faced_open_goal']):>7} {pct(c['regain_to_shot']):>11} {num(c['regain_to_shot_s']):>5} "
            f"{num(c['danger_s_per_match']):>6} {pct(c['free_kick_to_shot']):>7} "
            f"{num(t['pass_progress_m']):>8} {pct(t['forward_pass_share']):>4}"
        )


def _print_ball_losses(b: dict, run_id: str) -> None:
    def top(counts: dict[str, int], k: int = 5) -> str:
        return ", ".join(f"{name} {c}" for name, c in list(counts.items())[:k]) or "none"

    print(
        f"\nBALL LOSSES (config_a side): {b['real_losses']} real ({b['real_losses_per_match']}/match; "
        f"{b['raw_turnovers']} raw MatchStats turnovers)"
    )
    print(f"  by kind:   {top(b['by_kind'])}")
    print(f"  fouls:     {top(b['fouls_by_rule'])}")
    print(f"  by tactic: {top(b['by_tactic'])}")
    rec = b.get("receptions")
    if rec and rec["passes"]:
        print(
            f"  passes:    {top(rec['by_outcome'])}; {rec['catch_rate'] or 0:.0%} of reachable caught, "
            f"missed median facing off {rec['missed']['median_facing_off_deg']} deg"
        )
    print(f"  full breakdown: replays/{run_id}/ball_losses.md")


if __name__ == "__main__":
    main()
