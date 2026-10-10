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
- **Repo root cleanup** — first decided not to move root scripts (~45 inbound citations; `1f542fd`), then overturned 2026-10-02 because the root had grown to a dozen scripts and `tools/` already held the other tooling: the tournament drivers and `match` are now `tools/evaluation/`, and `debug_match` and `repro_from_replay` are in `tools/` (`elo` and `plot_elo` were removed 2026-10-10). `main.py`, `conftest.py` and `dashboard_server.py` stay at the root (pinned by pixi tasks, pytest and the README); `start_test_env.sh` was removed 2026-10-09 (unused; its three commands are in `docs/setup_external.md`).

## Open

1. **Current stall count (2026-10-01, `fpp`, `e83a7466`): 0/231 matches** in the full strict
   65s round-robin (`replays/tournament_20261001_094103/`), none flagged by the possession
   backstop; 93% of restarts reach NORMAL_START. `utama_core/scenario_bench/banks/bank_v7.json` is
   harvested from it. Earlier stall families (overload-slot COMMITTED_FROZEN, DIRECT_FREE
   restart stalls, low_block's PassAndShoot holding the ball) are fixed; see `git log`.
   Confirm any stall fix against the full round-robin: small-subset re-runs have repeatedly
   overstated fixes.

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
     metrics. **Built (v1):** `start.py`, 4 hand-authored anchors,
     `scenario_harvester.py` (restart transitions, trust gate `stall_events == []`),
     `dynamic_screen.py`, `scenario_scorer.py`, `tools/scenario_bench.py`. 2026-09-24: TURNOVER
     counts real losses only (raw turnovers are mostly nearest-robot flicker), stalls are a
     flag; `--repeats K` jittered starts give the seed noise rsim's determinism hid (the
     screen's default pool was the champion itself, so every live scenario came out
     DETERMINED); `--against-results` compares against an earlier run's JSON, for A/B
     across commits. Banks in `utama_core/scenario_bench/banks/` are committed (v3 onward; v1/v2 predate
     the rulebook referee fixes and stay local). **Bank v4** (173 scenarios) = v3 (from
     RR `tournament_20260928_132051`) + RR `tournament_20260928_125725`, each harvested,
     screened (informative + noisy kept), then merged dropping near-duplicate starts (same
     family, perspective and command; ball within 0.10 m, every robot within 0.15 m). v3 alone
     held 25 such duplicates of 142, mostly kickoff formations; a second run adds ~50 new
     scenarios, mostly free kicks, since kickoffs repeat and harvested open-play counters are
     mostly dead. Replays of a round-robin are only needed until it is harvested. v4 at
     `4b40f3c2` (press_and_pass vs low_block, 3 repeats): self vs self 173/173 identical; passes
     aimed 10° off: mean -0.143, se 0.045, t = -3.19 (57 worse, 26 better), all from the free
     kicks (v3: t = -2.23). **Bank v5 and no dynamic screen (2026-09-28):** the v1-v4 numbers above predate
     `c27bfad7` (hysteresis state leaked between teams and matches). The dynamic screen was
     removed: it varied the opponent with the candidate fixed, while an A/B varies the
     candidate. Of the starts it dropped as determined, the candidate's strategy changed the
     outcome in 24/40, and passes aimed 10° off moved them as much as the kept ones (dropped
     t = -6.9 on 489, kept t = -3.9 on 339; press_and_pass vs low_block, 1 run each). It also
     counted starts that failed to set up (robot past the field line) as neutral. v5 = every
     start of `tournament_20260928_132051` with `--open-play 2`, near-duplicates and
     out-of-field robots dropped: 846 starts, built in 3 min, one pass 11 min at 15 workers.
     press_and_pass vs low_block, 1 run each: passes aimed 10° off, t = -7.56 over 845. With a 10 s
     horizon instead of 20 s the pass takes 7-8 min but the same weakening is not seen (t = +0.76):
     most of its effect comes after 10 s.
     **Bank v1**
     (31 scenarios; rebuild with `--harvest-from` + `--dynamic-screen` + `--save-bank`): harvested
     from `tournament_20260924_124033` (692 restart scenarios from 222 stall-free matches), a
     stratified 72 screened (3 opponents x 3 jittered starts, 20s): 14 informative, 17 noisy
     (kept), 22 determined, 19 dead (dropped; 16 of the 20 FORCE_START restarts were dead).
     Seed noise and policy spread are about the same size (~0.1-0.25 ordinal units), so a
     real difference needs many scenarios; read the stderr. **Not built:**
     harvesting from a real post-fix calibration run (only synthetic fixtures so far), the
     lost-play weakness subset, event-triggered open-play harvesting, bank versioning/lifecycle.
   - *Outer loop, slow half (ladder, acceptance gate):* candidate vs a frozen reference pool
     of 4-5 strategies, both colours, K fuzz seeds, Elo anchored to the pool; full 600s
     matches only vs the top of the pool. **Not built.**
   - *Metrics — derive, don't invent:* `MatchStats` has both-sided turnovers, completed
     passes, attacking-third entries, possession-under-pressure, restart-to-entry. The first
     correlation study (`tools/metric_correlation.py`, `git show 24ef2c3f:benchmark_results/
     metric_correlation_20260903.md`) found turnovers/passes/entries valid and reliable;
     possession and robot motion carry no signal; `ball_travel_m` is not a quality signal.
     Caveat: raw `turnovers` is mostly nearest-robot flicker on contested balls (1990 raw vs
     1090 real losses over 231 matches, `git show 24ef2c3f:benchmark_results/turnover_breakdown_20260923_211717.md`);
     prefer `summary.json["ball_losses"]["real_losses"]`.
     Goodhart guard: if the bench improves and the ladder doesn't, retire the proxy.
   - *Compute discipline:* paired comparison on common seeds, sequential stopping, short
     sampled horizons over long matches.
   - **Not built:** one `evaluate <strategy> --budget` entry point (contracts + bench at low
     budget, ladder at high) that also writes `docs/strategies.md`.
   - Build order: calibration tournament → bench on harvested states → ladder.

2a. **Strategy evaluation v2 (direction agreed 2026-10-09; ladder built).** Builds the ladder
    above and settles how matches vary.
    - *Ladder:* a candidate plays a fixed reference pool (the current top 4-5) in several
      sampled worlds, each world played twice with the teams swapped so luck cancels, both
      kickoffs covered, stopping early once the result is clear; Elo anchored to the pool. The
      full round-robin stays as the occasional full refresh after shared-code changes. Pure
      Elo matchmaking over the whole league is not the plan: strategies counter each other, and
      the round-robin's results table is what shows that.
      **Built (2026-10-09):** `tools/evaluation/ladder.py`. Until worlds vary, a pairing has 8
      distinct matches (4 side/kickoff settings, each mirrored); it stops at 4 or more once wins
      minus losses reaches 3 either way or can no longer change sign. Not built: Elo anchoring
      (points per match against the pool is the read for now) and spot-checks of reused records.
    - *Sampled worlds instead of restart fuzzing:* variation comes from realistic imperfection,
      seeded so a world is reproducible: vision noise and dropped detections (`rsim_noise`,
      `rsim_vanishing` already exist), then kick speed and direction spread and command delay
      or lost packets (Python, in the sim's controller and `standard_ssl.py`), then per-robot
      profiles once 10c has measurements, then dribbler loss (likely `vendor/rSim`). Both teams
      face the same world with mirrored profiles. Ranges stay modest and written down until
      measured. The distribution is fixed by the evaluator and out of reach of strategy
      branches, so it is not something to tune toward. `RestartFuzzingReferee` stays as a
      stall-testing tool, not an evaluation input.
    - *Names:* one match runner with the schedule (round-robin, ladder) and match settings
      (sides, kickoffs, world seed) as options, replacing round-robin / full-match tournament /
      Elo as separate tools. `full_match_tournament.py` (both kickoffs) and `tools/elo.py`,
      `plot_elo.py` are retired only once the ladder covers them.
    - *Match cache:* a match's key (`replay/fingerprint.py` `match_key`) hashes the exact bytes
      and paths of everything `tools/evaluation/match.py` imports, data files next to
      them (not `.md`), both strategies' modules, and the match settings. New tools that call
      `run_match`, and new settings added to the key only when switched on (as `fuzz_seed` is),
      keep the cache. Renaming, moving or editing `match.py` or anything it imports
      (even a comment), or adding sim noise, reruns every match. So: build the ladder and world
      settings as additions first, and do the renames in the same batch as the next change that
      forces a full rerun anyway (the sim noise).
    - *Order:* write the ladder (additions only, done) → switch on vision noise and dropouts as world
      settings → kick spread and command delay, together with the naming unification, then one
      full round-robin → per-robot profiles after calibration.

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
    identical code, and a re-run flipped 3 of them back (load average ~30). Not reproduced
    2026-10-10: a 60 s split_shape vs high_press match gave identical stats under
    `PYTHONHASHSEED` 0, 1 and 2 and in 14 concurrent copies, and nothing in the rsim path reads
    the wall clock or unseeded randomness (the restart fuzzer seeds its own). Since-fixed rsim
    state bugs (`vendor/rSim/FORK_NOTES.md`, reused-sim reset) may have been the cause. Every
    `--reuse` round-robin now keeps its spot-check mismatches in `summary.json`
    (`reuse.mismatches`): a non-empty list there is the evidence to chase.

10b. **Ball-holding contract for every `Tactic`.** Most fouls fixed on 2026-09-24
    (`ShadowAndMark`, `PressAndContain`, `DecoyAndOverload` lure, keeper) were one pattern: a
    tactic written for its main job with no branch for "my robot now has the ball", so it held
    or carried until `ExcessiveDribbling`/`KeeperHeldBall` fired or both teams froze on the
    ball. Idea: one parametrized test over all tactics that puts the ball on each assigned
    robot in a few standard spots and checks the tactic releases it. Decisions still open:
    - *Depth:* single-tick check (first command is kick/pass, turn-to-kick, or a tracked
      carry) vs a short kinematic rollout that proves release before the carry limit and the
      keeper hold time. Leaning single-tick first; rollout only if something slips past.
    - *What counts as a valid response per tactic:* pass vs clear vs carry, and who decides
      (the tactic, or a shared default like `skills/src/kick_upfield.py`).
    - *Where the guarantee lives:* per-tactic branches (current approach), or an engine-level
      safety net that takes over a robot holding the ball too long — the latter is a new
      kernel concept, so it needs a concrete case the per-tactic approach can't handle.
    - *Setup for role-based tactics* (passer/receiver, committed slots): which robot gets the
      ball, and in which phase.
    - *Rule constants:* tactics currently copy limits (`CARRY_LIMIT_M = 0.8`) instead of
      importing them from the referee rules; decide whether to share one source.
    Expect each failing tactic to be a real fix with its own regression test.

10c. **Real-robot profiles and calibration.** Each physical robot differs (top speed,
    dribbler grip, kicker strength) in ways hardware can't fix soon; software should
    correct what it can and use the rest. No measurements yet (2026-10-09), so nothing per
    robot is built: this entry is the plan.
    - *Known now:* the kicker is fixed power and will stay so for now (hardware team,
      2026-10-09), so the sim gets no variable-kick option. Real robots are capped at
      `MAX_VEL=1` m/s as a safety limit, not their top speed; rsim and grSim run at 2 m/s
      (`config/robot_params.py`). The radio packet has 4-bit kick and chip power fields that
      we always send as full (`real_robot_controller.py`, `kicker_byte`): ask whether the
      firmware reads them.
    - *What exists:* `StrategyRunner`'s `{yellow,blue}_vision_to_cmd_mapping` (vision ID to
      firmware command ID, real mode only) is the robot roster. It is validated (one entry
      per expected robot, integers, no command ID used by both teams on a shared transmitter,
      every observed vision ID covered: `game_gater.py`) and filters vision to those IDs.
      `*_trusted_ir_robots` (robots whose ball sensor is trusted, the rest infer possession
      from vision) is the only per-robot capability today. There is no checked-in roster
      file: whoever writes the run script passes the mapping. The command ID is fixed in the
      robot's firmware, so it is the key a profile would use.
    - *What can differ per robot:* driving (top speed, acceleration and braking, turning,
      drift from a weaker motor, command delay, battery sag over a match); the ball (dribbler
      grip while driving and turning, catching a pass, kick speed and its spread, kick
      direction error, kicker recharge time, chip distance and height, ball sensor); other
      (radio packet loss, breakdowns and substitutions).
    - *Agreed:* correct in the real-robot controller what can be corrected (speed error,
      drift, delay), capping the team to its weakest robot where needed, so strategies see
      identical robots. Expose to tactics only what can't be corrected (kick strength, grip,
      catching). Tactics ask about abilities ("who dribbles best", "where does this robot's
      kick stop"), never name a robot ID. Profiles are measured, fixed for a run and out of
      reach of strategy branches, so they are not something a search tunes.
    - *Open, not yet:* one profile per match day vs updated while playing; kick-to-kick
      spread in the sim (more realistic, but noisier round-robins).
    - *Calibration routine (to write before measuring day):* per robot, at full battery on
      the competition carpet: commanded vs measured speed at a few speeds, acceleration and
      braking from vision, straight-line drift over 3 m, turn rate; ten kicks (speed from
      vision, direction error, roll distance) and ten chips; recharge time between kicks;
      a dribble course at increasing speed until the ball is lost; ten passes received.
      Output: one record per command ID.
    - *Before measurements:* code that assumes kick or speed numbers should derive them from
      `RobotParams` (e.g. `ClearBallTactic` assumes a 4.5 m clearance where a fixed-power kick
      rolls ~17 m), so measured values change one file. No per-robot profile type until
      measurements show robots differ enough to matter.

10d. **Vision filter for real cameras.** Found 2026-10-09 reviewing `data_processing/`; rsim
     (noise off) never exercises any of it. Needs logged SSL-Vision data from the real field
     to tune, so it waits on the hardware team like 10c.
     - *Velocity:* `KalmanFilter` tracks position only; `VelocityRefiner` differentiates
       consecutive filtered positions and that velocity feeds the next prediction. With the
       current noise settings (process noise twice the measurement noise) the steady-state
       gain is about 0.73, so positions are barely smoothed. An estimate: 1 cm vision noise
       gives roughly ±0.5 m/s velocity jitter at 60 Hz. The usual SSL design is one filter with
       position and velocity in its state (constant velocity for robots; for the ball, rolling
       friction, plus a chip/flight model later). The `VelocityRefiner` note that smoothing
       velocity "broke control loops" was measured in noiseless rsim.
     - *Lost objects:* a vanished ball or robot is predicted at its last velocity forever,
       with no friction and no give-up time. The ball vanishes most often under a dribbling
       robot, where it should stay at the dribbler. Vanished robots relate to substitutions,
       issue #107.
     - *Detections:* `CameraCombiner` ignores SSL-Vision confidence, so a low-confidence
       false detection with a robot's ID is averaged into the real one.
     - *Noise settings:* 1 cm and 5° match rsim's noise generator, not the real cameras.
     - *First step once logs exist:* replay a recorded vision log through `PositionRefiner`
       and compare velocity jitter and lag against a constant-velocity filter, before
       changing anything.

11. **Deferred, revisit only when forced** (minimalism):
    - Shared `Sticky`/hysteresis helper beyond `shared/tolerance.py` — existing instances
      differ in shape.
    - `@pytest.mark.engine` marker for kernel tests — wait for real CI bottleneck data.
    - `TickContext` responsibilities beyond `motion_controller`/`match_log`.
