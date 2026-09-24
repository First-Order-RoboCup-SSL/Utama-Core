# Roadmap / TODO

Larger open workstreams. Each entry: what, why, status, pointer. Investigation history lives
in git log (`git log --grep=<topic>`) and commit messages, not here — when an item is
resolved, replace it with a one-line pointer under "Done".

## Done (one-line pointers)

- **More tactics (first pass)** — `press_and_contain`, `give_and_go`, 5 example configs — `dd73bbc`.
- **BT/py_trees removal** — `AbstractStrategy` rewritten kernel-native — `087ee4b`, `960662c`, `48affd6`.
- **Tournament scoreless-draw debugging** — rSim kick direction, `FastPathPlanner`, `SwitchOfPlayTactic`, `DefenseTactic` foul loop, `TwoDPID` braking — `27fd36f`, `e11094a`, `78fce73`.
- **Motion-controller discontinuity handling** — PID auto-reset on orientation target jump — `91100ff`, `87cf250`, `036077c`.
- **CustomReferee gaps** — double touch, ball speed, full-episode `reset()`, `set_debug_status` rename (see `docs/custom_referee.md`).
- **Strategy-computation perf pass** — ~3x end-to-end — `3086337`, `0cf1e17`, `679e8cd`, `b79b863`, `b98cd29`, `4493002`.
- **Goalkeeper overshoot** — `a4df59e`, `ba59c9e`.
- **Stuck-match root causes** — `PressAndContainTactic` loose ball, `GiveAndGoTactic` hop timeout — `d6b3ff1`.
- **Defense-area retrieval stall** — `7a8e717`; trajsample equivalent in `TrajectorySamplingPlanner._enemy_defense_area_retrieval_exempt`.
- **Dashboard rebuild** — Live/Replay/Tournament views — `c3c389e`.
- **Touchline avoidance + placement-into-defense-area stall** — `3513c59`.
- **SSL §8.3/8.4 audit + 7 referee rules** — `434ab29`.
- **Restart formations read real goalkeeper id; `referee_overrides` hook** — `7cd1f61`.
- **`FORCE_START` out of a `STOP`/`HALT` pause is a barrier reset** (ball may have moved) — `46e962b`.
- **`robosim` stdout corrupting the JSON pipe** — `f613411`.
- **`score_pass_setup` pass scoring** and **GiveAndGo permanent ball-lock** — `dd14f79`.
- **Restart safety bugs (testing_gaps #7, #8)** — `ccb172b`.
- **Ball-contest deadlock** — treated as a `PushingRule` no-fault case, not a planner fix — `a3f3795`.
- **`custom_referee` missing-`designated_position` audit** — `2a8c03f`.
- **trajsample liveness fixes** — escape grace `2e53f3e`; target-jitter tolerance `5183ed1`; brake scales planned velocity; stale intermediate target; restart-scoped priority blocking `9757814`; two-segment candidate instability `4b701ae`; teleport-settle speed + 10s ball-placement timeout `43d41af`; four `COMMITTED_FROZEN` causes `70cb5c6`; `DefenseTactic` carrier clearing; `DirectFreeOursStep`/`BallPlacementOursStep` sticky kicker/placer.
- **Sumatra-fidelity audit, top 3 findings** — acceptor leniency, escaping-grace clearance, full-window priority re-check.
- **Tournament gate tooling** — `--strict`, `--stop-at-first-stall`, `--fuzz-restarts SEED` (`405693c`).
- **`metric_correlation.py` reads `.npz`**.
- **Repo root cleanup** — decided not to move root scripts (~45 inbound citations; `conftest.py`, `main.py` and `tournament_lib` are pinned) — `1f542fd`.

## Open

1. **Current stall count (2026-09-23, `fpp`, `cb6460a`): 8/231 matches** in the full strict
   65s smoke round-robin (`replays/tournament_20260923_211717/`), plus 2 flagged by the
   possession backstop (`overload_flow`/`score_aware_zone_flow` vs `split_shape`: 100%
   possession, 0.57m ball travel); 41/231 decisive. Families:
   - 4 `COMMITTED_FROZEN`, all in the `overload` slot (`DecoyOverloadTactic`, 3 of them
     `high_line_zone`, onset t=56s). Under `trajsample` this family was traced to a receiver
     boxed in by two enemies (`70cb5c6`); that it also shows under `fpp` suggests the tactic,
     not only the planner. Unverified.
   - 4 `RESTART_STALL` at `DIRECT_FREE_*` (`give_and_go_solo`/`high_line_zone`,
     `high_line_zone`/`split_shape`, `split_shape`/`switch_of_play`, `three_slot`/`zone_fluid`).
     Two causes fixed 2026-09-24 (single-match verified, not yet re-measured in a round-robin):
     FPP detour subgoals inside the enemy box (`e168f45`) and the placement teleport landing
     on a robot (`dd00ae1`).
   `trajsample` not re-measured. Confirm any stall fix against the full round-robin —
   small-subset re-runs have repeatedly overstated fixes.

2. **Outer-loop strategy evaluation.** Goal: agents iterate strategies against evals without
   humans watching replays. Win rate/Elo is the objective but too sparse and too expensive to
   be the only signal (most 65s matches are draws; rsim is deterministic, so a repeat adds
   nothing unless restarts are fuzzed). Three tiers:
   - *Inner loop (tests, binary, blocks merge):* tactic contracts, kernel invariants, zero
     `StallEvent`s across N seeded fuzzed matches. **Built:** stall watchdog + `--strict`,
     commitment deadline, restart fuzzer, pure `Game` builder. **Not built:** CI job running
     the strict seeded stall gate.
   - *Outer loop, fast half (scenario bench, numbers vs a committed baseline, non-blocking):*
     seeded start state + 15-30s horizon, scored as paired differentials by calibrated proxy
     metrics. **Built (v1):** `bench_scenario.py`, 4 hand-authored anchors,
     `scenario_harvester.py` (restart transitions, trust gate `stall_events == []`),
     `dynamic_screen.py`, `scenario_scorer.py`, `tools/scenario_bench.py`. **Not built:**
     harvesting from a real post-fix calibration run (only synthetic fixtures so far), the
     lost-play weakness subset, event-triggered open-play harvesting, bank versioning/lifecycle.
   - *Outer loop, slow half (ladder, acceptance gate):* candidate vs a frozen reference pool
     of 4-5 strategies, both colours, K fuzz seeds, Elo anchored to the pool; full 600s
     matches only vs the top of the pool. **Not built.**
   - *Metrics — derive, don't invent:* `MatchStats` has both-sided turnovers, completed
     passes, attacking-third entries, possession-under-pressure, restart-to-entry. The first
     correlation study (`tools/metric_correlation.py`, `benchmark_results/
     metric_correlation_20260903.md`) found turnovers/passes/entries valid and reliable;
     possession and robot motion carry no signal; `ball_travel_m` is not a quality signal.
     Caveat: raw `turnovers` is mostly nearest-robot flicker on contested balls (1990 raw vs
     1090 real losses over 231 matches, `benchmark_results/turnover_breakdown_20260923_211717.md`);
     prefer `summary.json["ball_losses"]["real_losses"]`.
     Goodhart guard: if the bench improves and the ladder doesn't, retire the proxy.
   - *Compute discipline:* paired comparison on common seeds, sequential stopping, short
     sampled horizons over long matches.
   - **Not built:** one `evaluate <strategy> --budget` entry point (contracts + bench at low
     budget, ladder at high) that also writes `docs/strategies.md`. Prerequisite: consolidate
     `arena_tournament.py` onto `tournament_lib.run_match` (it still keeps its own
     `_run_match` for per-tick instrumentation hooks).
   - Build order: calibration tournament → bench on harvested states → ladder.

3. **`BangBang1D` defects** — required-overshoot and `v0 > v_max` cases produce
   discontinuous trajectories (xfail-pinned in `utama_core/tests/motion_planning/
   implementation/bang_bang_edge_cases_test.py`). A correct fix (`b26a550`) was reverted (`90d068c`) because
   physically correct trajectories let mutually blocked robots stall at kickoff (231/231).
   Re-apply only together with an explicit yield/escape result for the all-candidates-collide
   state.

4. **trajsample architecture follow-ups** (from a TIGERs 2024 comparison + Sumatra source
   audit; validate each with `tools/motion_planning_benchmark.py`, see
   `docs/motion_planning_comparison.md`), roughly in order:
   - Directional-tube enemy obstacle (`obstacles.py`) — attempted and reverted (regressed
     `mirror_swap`, doubled `static_slalom`). Next attempt must stay at least as conservative
     as the circle at small horizons.
   - `Trajectory2D` drops transverse velocity at trajectory start (`bang_bang.py`); every
     trajectory ends at zero velocity. Want a concrete replay failure before starting.
   - Fixed robot-id priority can make the wrong robot yield; no first-class "yield" result.
   - Collision sampling is adaptive-timestep, not swept; intermediate targets are purely
     random (hurts reproducibility in corridors); committed-trajectory reuse could validate
     more.
   - Sumatra audit findings #4-7: untriaged.
   - Last: a jerk-limited generator (e.g. Ruckig).

5. **FastPathPlanner convergence stall (`test_mirror_swap`, xfail)** — two wing robots stall
   ~0.53m short in a mirrored 6v6. Genuine local minimum. DWA solves this scenario but costs
   2-3x compute and fails others; no scheme wins every benchmark scenario, so the default
   (`fpp`) is unchanged.

6. **Referee testing gaps** — see `docs/testing_gaps.md`. Open: mypy/pyright adoption (on
   hold); `ball_placement_interference` has never fired live (not treated as a bug).

7. **Cross-tactic ball convergence** — nothing checks whether another assigned robot is
   already converging on the same ball; fixed twice locally (`a59a8e5`, `0e510a0`/`ae6a6f3`).
   Generalize only if a third instance appears.

8. **grsim as a CI/tournament environment.** Unverified whether grsim installs on a GitHub
   runner or can run faster than real time headless. CI hardcodes `--ignore-glob "**/*grsim*"`.

9. **Geometric intention overlays for the Replay tab** — e.g. `ShadowAndMarkTactic._assign_marks()`
   already computes `{marker: opponent}` each tick and discards it. Tactic side is small via
   `MatchLog.trace_if_changed()`; `field_canvas.js` has no line/arrow primitive yet.

10. **More tactics from real football vocabulary** (formations, set plays, pressing schemes)
    rather than variations on existing ones.

10a. **rsim matches are not fully run-to-run deterministic.** Found 2026-09-23: in a
    56-match `--fuzz-restarts 1` run, 4 `press_and_pass` matches gave different results on
    identical code, and a re-run flipped 3 of them back. Paired-seed evaluation (item 2) and
    `--stop-at-first-stall` both assume determinism. Unexplored; first suspects are
    `PYTHONHASHSEED` (set/dict iteration order) and wall-clock-dependent code under machine
    load (load average was ~30).

11. **Deferred, revisit only when forced** (minimalism):
    - Shared `Sticky`/hysteresis helper beyond `shared/tolerance.py` — existing instances
      differ in shape.
    - `@pytest.mark.engine` marker for kernel tests — wait for real CI bottleneck data.
    - `TickContext` responsibilities beyond `motion_controller`/`match_log`.
    - `start_test_env.sh` has no inbound references and launches a gitignored `AutoReferee/`
      — relic or live hardware script? Needs someone with the hardware.
