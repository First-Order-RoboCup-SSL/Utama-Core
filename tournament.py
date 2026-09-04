"""tournament.py — Round-robin every `kernel_strategy.py` config against every other.

Run:
    pixi run python tournament.py

What this does
--------------
Headless rsim, no external process required. For each distinct pair of the
`build_*_kernel_strategy` factories in `utama_core.strategy.kernel_strategy`,
runs one `StrategyRunner` match (6v6: 1 goalkeeper + 5 outfield robots per
side — enough for every factory's minimum, including the `min_attack=2`
configs and `build_three_slot_kernel_strategy`'s three concurrent slots),
steps it for `MATCH_DURATION_SECONDS` of sim time, and records the final
score from `CustomReferee`'s scoreboard.

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
`config_a` (yellow, defending/attacking the right side per `run_match`'s
hardcoded `my_team_is_yellow=True, my_team_is_right=True`) — instead of
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

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.restart_fuzzer import RestartFuzzingReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_writer import ColumnarReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_OUTFIELD = 5  # + 1 goalkeeper per side
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
# 60s of intended play, +5s for a real PREPARE_KICKOFF_YELLOW ceremony
# (prepare_duration_seconds=3.0 in the "simulation" profile, plus the kicker's
# walk to the centre circle — observed ~5s total; see run_match's
# referee_initial_command) so a "60s" tournament match still gets 60s of live
# play rather than 60s minus ceremony overhead.
MATCH_DURATION_SECONDS = 65.0
TICKS_PER_SECOND = 60  # matches rsim's default step rate

# build_default_kernel_strategy is excluded from the auto-discovered catalog:
# despite the name, it isn't a competitive team — it's the kernel's minimal
# single-tactic smoke-test scaffold (see its docstring), used across the test
# suite with as few as zero outfield robots, and the arena-strategy stats
# investigation confirmed it plays a real match with 3 of 5 outfield robots
# never issued a command ("zombie" robots, 0.0 motion all match). Fixing that
# would still only produce a deliberately-minimal team, not a useful
# comparison point. build_tiki_taka_kernel_strategy — the strongest, most
# complete team by the same stats investigation (live-state posture,
# possession-backed wins, no losses) — is the de facto baseline other configs
# get judged against instead; it needs no special-casing here since it's
# already just another entry in the catalog.
_CONFIG_NAMES = [
    name
    for name in dir(kernel_strategy)
    if name.startswith("build_")
    and name.endswith("_kernel_strategy")
    and callable(getattr(kernel_strategy, name))
    and name != "build_default_kernel_strategy"
]


@dataclass
class MatchResult:
    config_a: str
    config_b: str
    score_a: int
    score_b: int
    stats: Optional[dict] = field(default=None, compare=False)

    @property
    def winner(self) -> str:
        if self.score_a > self.score_b:
            return self.config_a
        if self.score_b > self.score_a:
            return self.config_b
        return "draw"


def _short_name(config_name: str) -> str:
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def run_match(
    config_a_name: str,
    config_b_name: str,
    run_dir: Optional[Path] = None,
    control_scheme: str = "fpp",
    fuzz_seed: Optional[int] = None,
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0),
) -> MatchResult:
    """Play one match. If `run_dir` is set, also records the full observability
    stack (structured intention log, aggregate stats, replay trail) under it —
    see `utama_core.engine.match_log`/`match_stats` and `utama_core.replay`.

    `control_scheme` is used for both sides (matching StrategyRunner's default
    of falling back to `control_scheme` when `opp_control_scheme` is unset) —
    this script exists to compare strategies against each other, not motion
    planners against each other (see `tools/motion_planning_benchmark.py` for
    that), so there's no need for the two sides to differ here.

    `fuzz_seed`, if set, swaps in `RestartFuzzingReferee` (see
    `utama_core/custom_referee/restart_fuzzer.py`) instead of plain
    `CustomReferee`, so this match's referee injects extra seeded-random
    legal restarts during live play — see `main()`'s `--fuzz-restarts`/
    `--fuzz-interval` flags.
    """
    build_a = getattr(kernel_strategy, config_a_name)
    build_b = getattr(kernel_strategy, config_b_name)

    strategy_a = AbstractStrategy(build_kernel_strategy=build_a(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_b(OUTFIELD_ROBOT_IDS))

    if fuzz_seed is not None:
        referee = RestartFuzzingReferee.from_profile_name(
            "simulation",
            seed=fuzz_seed,
            interval_s=fuzz_interval_s,
            n_robots_yellow=N_OUTFIELD + 1,
            n_robots_blue=N_OUTFIELD + 1,
        )
    else:
        referee = CustomReferee.from_profile_name(
            "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
        )
    # "simulation" profile's kickoff_team defaults to "yellow", and config_a is
    # always yellow (my_team_is_yellow=True below) — so config_a always kicks
    # off. Without this, StrategyRunner defaults sim-mode matches to
    # FORCE_START (both teams released simultaneously at a ball equidistant
    # from mirror-symmetric formations), which — root-caused 2026-08-23, see
    # docs/strategies.md's "Known open bugs" — lets sub-millimetre rsim
    # physics noise decide who's "closer to the ball" and cascade into a
    # different match. A real PREPARE_KICKOFF_YELLOW ceremony avoids that
    # simultaneous-race condition entirely.
    initial_command = RefereeCommand.PREPARE_KICKOFF_YELLOW

    match_tag = f"{_short_name(config_a_name)}_vs_{_short_name(config_b_name)}"
    extra_kwargs = {}
    if run_dir is not None:
        extra_kwargs["match_log_path"] = str(run_dir / f"{match_tag}.intentions.jsonl")
        extra_kwargs["stats_path"] = str(run_dir / f"{match_tag}.stats.json")
        # replay_name is relative to REPLAY_BASE_PATH, not run_dir, since replays
        # live under a fixed replays/ root — nest it under the same tournament
        # subdirectory so the two stay next to each other on disk.
        #
        # Columnar (.npz) rather than pickle (.pkl): ~13 MB/match in the old
        # format vs. a fraction of that here, and every reader a tournament
        # run's replays are actually fed through — `load_frames_in_range`
        # (`replay_player.py`, used by `render_window`) and
        # `find_stuck_windows` (`stuck_detector.py`) — already dispatches on
        # `.npz` vs `.pkl` by extension, so nothing downstream of a
        # tournament run breaks. Only the interactive `play_replay`/
        # `get_latest_replay_name` CLI helpers in `replay_player.py` still
        # hardcode `.pkl`; tournament.py doesn't call either.
        extra_kwargs["replay_writer_config"] = ColumnarReplayWriterConfig(
            replay_name=f"{run_dir.name}/{match_tag}", overwrite_existing=True
        )

    runner = StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        referee_initial_command=initial_command,
        control_scheme=control_scheme,
        **extra_kwargs,
    )

    try:
        for _ in range(int(MATCH_DURATION_SECONDS * TICKS_PER_SECOND)):
            runner.step_once()
        ref_data = runner.my.game.referee
        score_a = ref_data.yellow_team.score
        score_b = ref_data.blue_team.score
        stats = runner.match_stats.finalize() if runner.match_stats is not None else None
    finally:
        runner.close()

    return MatchResult(
        config_a=config_a_name,
        config_b=config_b_name,
        score_a=score_a,
        score_b=score_b,
        stats=_stats_to_dict(stats) if stats is not None else None,
    )


def _stats_to_dict(stats) -> dict:
    """`MatchStats.__dict__`, but with `stall_events` (a list of `StallEvent`
    dataclasses) turned into plain dicts so the result round-trips through
    `json.dump` in `summary.json` — mirrors `MatchStats.to_json`'s own
    per-field shape rather than introducing a second serialization scheme.
    """
    d = dict(stats.__dict__)
    d["stall_events"] = [
        {
            "kind": e.kind,
            "sim_time": e.sim_time,
            "tick": e.tick,
            "referee_command": e.referee_command,
            "duration_s": e.duration_s,
            "tactic_ids": list(e.tactic_ids),
            "robot_ids": list(e.robot_ids),
        }
        for e in stats.stall_events
    ]
    return d


def main() -> None:
    # Optional CLI args: config names (with or without the `build_`/
    # `_kernel_strategy` wrapping) to run instead of the full auto-discovered
    # catalog — useful for a quick check of one or two configs without
    # waiting on every pair, e.g. `python tournament.py default
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
    args = sys.argv[1:]
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
    # injections when `--fuzz-restarts` is on. Default 25-45s: over a 65s
    # match this means one or two injections, not the 8-20s range in
    # RestartFuzzingReferee's own docstring, which is a stress-test example,
    # not a sane default for a normal round-robin match length.
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0)
    if "--fuzz-interval" in args:
        idx = args.index("--fuzz-interval")
        fuzz_interval_s = (float(args[idx + 1]), float(args[idx + 2]))
        args = args[:idx] + args[idx + 3 :]

    if args:
        requested = set(args)
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
        if len(config_names) < 2:
            raise SystemExit("Need at least 2 configs to play a round-robin.")
    else:
        config_names = _CONFIG_NAMES

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

    print(f"Round-robin: {len(config_names)} configs, {len(pairs)} matches" + (" (both sides)" if both_sides else ""))
    print(f"{N_OUTFIELD + 1}v{N_OUTFIELD + 1}, {MATCH_DURATION_SECONDS:.0f}s sim time per match, headless rsim")
    if no_save:
        print("--no-save: not recording replay/intention-log/stats for this run")
    else:
        print(f"Recording to replays/{run_id}/ (per-match replay, intention log, stats)")
    if not sequential:
        default_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = min(len(pairs), max_workers_override or default_workers)
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
    if stalled_matches or backstop_matches:
        print("\nSTALLS:")
        for r, events in stalled_matches:
            for e in events:
                tactic_str = f" tactics={e['tactic_ids']}" if e["tactic_ids"] else ""
                print(
                    f"  {_match_tag_of(r):<50} {e['kind']:<17} onset t={e['sim_time']:.1f}s "
                    f"referee={e['referee_command']}{tactic_str}"
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
        "config_names": sorted(config_names),
        "control_scheme": control_scheme,
        "match_duration_seconds": MATCH_DURATION_SECONDS,
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
            }
            for r in results
        ],
        "standings": {name: {"wins": wins[name], "draws": draws[name]} for name in config_names},
        "stalled_match_count": len(stalled_matches),
        "possession_backstop_match_count": len(backstop_matches),
    }
    if run_dir is not None:
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results + stats: replays/{run_id}/summary.json")

    if strict and (stalled_matches or backstop_matches):
        raise SystemExit(
            f"--strict: {len(stalled_matches)} match(es) with stall events, "
            f"{len(backstop_matches)} match(es) flagged by the possession backstop"
        )


if __name__ == "__main__":
    main()
