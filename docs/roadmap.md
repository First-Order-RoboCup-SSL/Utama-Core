# Roadmap / TODO

Larger open workstreams. Each entry: what, why, status, pointer. Investigation history lives
in git log (`git log --grep=<topic>`) and commit messages, not here — when an item is
resolved, replace it with a one-line pointer under "Done". Open items are numbered in order:
a resolved one leaves the list and the rest move up, so a number is not a stable name — cite the
entry by its title, not its number.

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
- **Strategy evaluation** — `tools/evaluation/evaluate.py` judges a candidate against the champion of a fixed, versioned pool (`tools/evaluation/pools/pool_v1.json`: champion, six opponents, three held-out opponents, seeded sim noise). Candidate and champion meet the same opponents under the same noise seed, decided on paired **goals** — P(candidate stronger) >= 0.99 promotes, <= 0.2 drops, undecided after 6 rounds; `analysis/ratings.py` fits strengths over the ledger (`replays/eval/ledger.jsonl`), which never replays a fixture it already has, and `--explain` draws where the robots stand. The situation bench and `ladder.py` were built, used and then retired — the bench because its estimate is biased per strategy, so it could not rank (Spearman -0.09, `docs/sim_realism.md` "Evaluation infrastructure"). How to use it: `docs/STRATEGY_DEVELOPMENT.md`, "Evaluating a strategy". `f48d4bad`, `1849d123`, `49f29ebf`.
- **Directional-tube enemy obstacle** — trajsample's moving-obstacle obstacle is a capsule expanded along the obstacle's velocity instead of an inflated circle (`utama_core/motion_planning/src/trajsampling/obstacles.py:132-149`) — `a1b8bdd6`.
- **FastPathPlanner `mirror_swap` stall** — the 0.52 m shortfall was the defense-area keep-out clamp, not a planner local minimum: back-row targets moved to |x| = 2.9 and the xfail went (`5f97199b`, in #141); #147 separately cleared the detour-side-flip corner case (`fastpathplanner_corner_test.py`) — `0ce4e2a2`. trajsample passes all 29 scenarios of the current table; fpp passed all 19 of the one before it.
- **Geometric intention overlays in the Replay tab** — `MatchLog.trace_if_changed` overlays drawn by `dashboard/static/field_canvas.js` with its `_drawArrow` primitive: `shadow_and_mark.marks`, `give_and_go.shot_lane`/`pass_target`, `defense.shadow_post`, `block_shape.screen_line`, `press_and_contain.press`, `clear_ball.clear_target`, `dribble.segment_target`, `decoy_and_overload.lure`, `committed_robot_ids` — `c47f7549`, `b6e69b44`.
- **Shared hysteresis helper** — `Sticky` in `shared/tolerance.py` (`fd87c7f6`), now used by `PassAndShootTactic`'s passer choice, `ClearBallTactic`'s target choice and `PressAndContainTactic`'s presser choice instead of per-site margins — `0ce4e2a2`. (`pickers.friendly_closer_to_ball` keeps its own tick-count hysteresis.)

## Open

1. **Stalls — latest full strict round-robin (2026-10-17, `replays/tournament_20261017_151052/`):**
   4 of 66 matches stalled, and 97% of 3232 restarts reached NORMAL_START. The kick-off that
   waited the referee's 15 s cap is fixed (`1731e904`); the three committed robots frozen about
   10 s (`COMMITTED_FROZEN` in `stall_incidents`) are not yet diagnosed. Confirm any stall fix
   against the full round-robin: small-subset re-runs have repeatedly overstated fixes. **Not built:** a CI job running the
   strict seeded stall gate — `.github/workflows/ci.yml` runs the test suite only.

2. **`BangBang1D` defects** — required-overshoot and `v0 > v_max` cases produce discontinuous
   trajectories, pinned as 78 strict xfails on the enumerated defect seeds
   (`utama_core/tests/motion_planning/implementation/bang_bang_edge_cases_test.py`;
   `trajsampling_correctness_test.py` pins the same defect). A correct fix (`b26a550`) was
   reverted (`90d068c`) because physically correct trajectories let mutually blocked robots
   stall at kickoff (231/231). trajsample has since gained Sumatra's all-candidates-collide
   fallback (`a1b8bdd6`), so re-applying the fix means validating it against that; there is
   still no first-class yield/escape result for the blocked state.

3. **trajsample architecture follow-ups** (from a TIGERs 2024 comparison; validate each with
   `tools/motion_planning_benchmark.py`, see `docs/motion_planning_comparison.md`):
   - `Trajectory2D` drops transverse velocity at trajectory start (`bang_bang.py`); every
     trajectory ends at zero velocity. Want a concrete replay failure before starting.
   - Priority: a ball carrier now outranks off-ball robots (`a1b8bdd6`), but the remaining
     tie-break is fixed robot id, which can make the wrong robot yield; still no first-class
     "yield" result.
   - Collision sampling is adaptive-timestep, not swept. (The other two parts of this entry are
     done: intermediate targets are a deterministic symmetric fan with a <=90 deg hemisphere
     clamp and last-direction stiction, and a committed trajectory is re-validated over a 0.5 s
     window.)
   - Last: a jerk-limited generator (e.g. Ruckig).

4. **Referee testing gaps** — see `docs/testing_gaps.md`. Open there: #4 mypy/pyright adoption
   (on hold). #6 stands as written for real matches, but our sim does reach the rule: a corner
   placement that trapped a defender inside the keep-out circle is pinned as a passing moment
   (`utama_core/tests/moments/placement_in_corner_traps_defender.json`, from
   `tournament_20261017_002337`).

5. **Cross-tactic ball convergence** — nothing checks whether another assigned robot is already
   converging on the same ball; the only guard is `DecoyOverloadTactic`'s
   `_teammate_already_has_ball()` (`utama_core/tactics/decoy_and_overload.py:99`). Found twice
   (`a59a8e5`, `0e510a0`/`ae6a6f3`); a third case (two tactics' robots converging on one ball)
   was diagnosed in `25888cce` and closed as a planner-level matter rather than by generalizing.
   Revisit only if it stalls a match again.

6. **grsim as a CI/tournament environment.** Unverified whether grsim installs on a GitHub
   runner or can run faster than real time headless. CI hardcodes `--ignore-glob "**/*grsim*"`
   (`ci.yml:41,43`).

7. **Match timeline in the dashboard.** Built 2026-10-21 (`dashboard/static/timeline.js`): the
    swimlanes of `tools/match_timeline.py` (referee band, each robot's tactic, possession, distance
    to the ball, events) under the Replay tab's controls, following the cursor, click or drag to
    seek; and in the Live tab, the last minutes up to now. Switch ticks, shots and a possession
    band (each situation's side) as in the tool. The Replay side panel's Play list replaces the raw
    event and intention logs: one row per situation (`goal_ledger.situations`: a restart going live
    or a side winning the ball), with its third, length, shots and how it ended (the stoppage's
    reason, or who won the ball and how); the raw intention log stays, folded. The slow analysis is
    kept beside the replay (`<stem>.timeline.json`) and ticks are sent as columns of whole numbers
    (a match: 4 MB gzipped), so a replay serves in ~1.5 s. Not yet: enemy lanes (no intentions are
    recorded for the opponent).

8. **More tactics from real football vocabulary** (formations, set plays, pressing schemes)
    rather than variations on existing ones. No new `Tactic` class has been added since this
    entry: the recent work refactors existing ones (`ShadowAndMarkTactic` now uses the `man_mark`
    skill, `ClearBallTactic`/`PressAndContainTactic`/`DecoyOverloadTactic` gained thresholds and
    exits).

9. **Ball-holding contract for every `Tactic`.** Most fouls fixed on 2026-09-24
    (`ShadowAndMark`, `PressAndContain`, `DecoyAndOverload` lure, keeper) were one pattern: a
    tactic written for its main job with no branch for "my robot now has the ball", so it held
    or carried until `ExcessiveDribbling`/`KeeperHeldBall` fired or both teams froze on the
    ball. Idea: one parametrized test over all tactics that puts the ball on each assigned
    robot in a few standard spots and checks the tactic releases it. `test_all_tactics.py` now
    covers the generic properties (declared tag, `is_committed()` bool, a command for every
    assigned robot) but not the ball-holding contract itself. Decisions still open:
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
    - *Rule constants:* tactics still copy limits (`CARRY_LIMIT_M` in
      `shared/pass_and_score_geometry.py`) instead of importing them from the referee rules;
      decide whether to share one source.
    Expect each failing tactic to be a real fix with its own regression test.

10. **Real-robot profiles and calibration.** Each physical robot differs (top speed,
    dribbler grip, kicker strength) in ways hardware can't fix soon; software should
    correct what it can and use the rest. One robot measured so far (2026-10-27); nothing per
    robot is built yet: this entry is the plan, with the to-do list below.
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
      `{yellow,blue}_robot_profiles` (`RobotProfile`: kicker, dribbler, ball sensor working
      or not, 2026-10-21) say what each robot can do; a robot without a kicker pushes the
      ball, one without a dribbler carries it slowly, one without a ball sensor infers
      possession from vision (`robot_profile.py`). These are on/off, not measured. There is no checked-in roster
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
    - *Measuring, built 2026-10-21:* every camera run records an SSL game log (raw vision,
      referee, radio bytes both ways: `GameLogWriter`); `examples/shooting_drill.py
      --calibrate` plays fixed moves (`tactics/scripted_moves.py`), and `tools/robot_calibration.py`
      measures loop delay, speed gain, acceleration, vision noise and kick speed from any such log.
      Checked on a synthetic robot (delay within 10 ms, acceleration within 15 %; turn
      acceleration only roughly, its ramp lasts a few frames) and on grSim; not yet on a real robot.
      Not yet done: the numbers feed nothing (the sim's latency, the controller), and drift,
      dribbling, passes and chips are not measured. First real robot 2026-10-27: delay 79 ms,
      turn gain 1.09, sideways acceleration ~1.8 against ~5 m/s² forward. 2026-10-28:
      `pixi run calibrate` (`examples/calibrate_robot.py`) runs it from the terminal, no
      dashboard: prompts for each set-up, prints each move, DONE, then the analysis and a
      `.calibration.json`; it now also holds, turns and drives with the ball and kicks 3 times
      (kick speed and direction off the heading, where the ball sits, ball sensor share, kept or lost).
    - *Calibration to-do (user, 2026-10-28; revisit in later sessions):*
      1. **Full robot profile**: one checked-in file per robot (keyed by command ID), written by
         `pixi run calibrate`, read by the controller, planner limits, kick aim, vision filter,
         sim noise and offline replay. Covers everything below plus the setup errors in
         [`field_log_replay.md`](field_log_replay.md) (tag yaw and centre offset, kicker offset).
      2. **Driving**: speed gain and acceleration per axis *and diagonally* (forward, sideways,
         45°): the top speed and acceleration a robot actually holds in each direction, not one
         `MAX_ACCELERATION`; braking; drift over 3 m; turn rate and turn gain.
      3. **Kicking**: ten kicks per robot: speed, direction offset and its spread, roll distance,
         recharge time; chips once used.
      4. **Dribbling**: hold while turning and driving at rising speed until the ball is lost:
         the fastest turn and drive that keep it. Decides whether a placer may grab a ball
         from any side and turn with it, instead of going round it first (step 5).
      5. **Receiving and grabbing**: the approach speed at which a robot still traps a ball,
         driving onto a still ball and receiving a rolling one (passes at a few speeds); whether
         it can turn on the spot smoothly with the ball at the end of an approach.
      6. **All robots at once**: several `--robot` in one run, the driving part in parallel lanes
         0.8 m apart (same moves at the same time, so no collisions), the ball part one robot at
         a time. The analysis already handles every robot in a log.
      7. **Use the numbers** (each a small change): sideways and diagonal acceleration limits in
         the planner, turn gain in the controller, delay prediction (plan from where the robot
         will be 79 ms on), kick aim by the kicker offset, Kalman measurement noise from vision
         noise, `contact_grace_s` from the sensor share. Sim-side use (latency, kick offsets,
         motion limits in `sim_noise`) replays the evaluation ledger: batch with a forced rerun.
         Robot-robot avoidance needs it most: on 2026-10-27 two robots met head-on at 1 m/s each
         (their placer running to the ball, ours clearing away from it along the same line) and
         touched; each planner saw the other, but 79 ms late and braking sideways at ~1.8 m/s².
         *Delay prediction done 2026-10-28:* `move()` plans from our robots' pose
         `RobotParams.CONTROL_LATENCY_S` (real: 0.079 s) ahead (`motion_planning/src/common/pose_ahead.py`).
         Checked in rsim given the calibrated robot (delay, gains and per-axis acceleration
         applied in the sim by a scratch harness): driving while turning, off-line drift 30 → 7 cm;
         motion benchmark collisions with the delay alone 7 → 3 (24 → 26 of 29 pass). With the
         sideways acceleration added it is still 24 of 29 with or without prediction.
         *Sideways acceleration, 2026-10-28:* `RobotParams.MAX_SIDEWAYS_ACCELERATION` (real 1.8),
         used by `trajsample` only; `fpp` plans paths, not motions, so it cannot use it. With the
         calibrated robot `trajsample` collides in 9 of 50 seeded close encounters against `fpp`'s
         25, and passes 29 of 29, but takes ~30 % longer in crowds
         ([motion_planning_comparison.md](motion_planning_comparison.md#on-the-calibrated-robot-2026-10-28)).
         *Open:* run the real robots on `trajsample` (`control_scheme="trajsample"`) in a field
         session and compare with `fpp` in the same drills; the sim evaluation stays on `fpp`.
    - *Calibration routine (the full plan):* per robot, at full battery on the
      competition carpet: commanded vs measured speed at a few speeds, acceleration and
      braking from vision, straight-line drift over 3 m, turn rate; ten kicks (speed from
      vision, direction error, roll distance) and ten chips; recharge time between kicks;
      a dribble course at increasing speed until the ball is lost; ten passes received.
      Output: one record per command ID.
    - *Before measurements:* code that assumes kick or speed numbers should derive them from
      `RobotParams` (e.g. `ClearBallTactic` assumes a 4.5 m clearance where a fixed-power kick
      rolls ~17 m), so measured values change one file. No per-robot profile type until
      measurements show robots differ enough to matter (to-do 1 above: build it once two
      robots' calibrations disagree by more than two runs of one robot do).

11. **Vision filter for real cameras.** Found 2026-10-09 reviewing `data_processing/`; rsim
     (noise off) never exercises any of it. Needs logged SSL-Vision data from the real field
     to tune, so it waits on the hardware team like "Real-robot profiles and calibration".
     - *Velocity:* `KalmanFilter` tracks position only; `VelocityRefiner` differentiates
       consecutive filtered positions and that velocity feeds the next prediction. With the
       current noise settings (process noise twice the measurement noise) the steady-state
       gain is about 0.73, so positions are barely smoothed. An estimate: 1 cm vision noise
       gives roughly +/-0.5 m/s velocity jitter at 60 Hz. The usual SSL design is one filter
       with position and velocity in its state (constant velocity for robots; for the ball,
       rolling friction, plus a chip/flight model later). The `VelocityRefiner` note that
       smoothing velocity "broke control loops" was measured in noiseless rsim. #144 only
       added a fast path to that differentiator (`a1b8bdd6`); the substance is unchanged.
     - *Lost objects:* a vanished ball or robot is predicted at its last velocity forever,
       with no friction and no give-up time. The ball vanishes most often under a dribbling
       robot, where it should stay at the dribbler. Vanished robots relate to substitutions,
       issue #107.
     - *Detections:* `CameraCombiner` ignores SSL-Vision confidence, so a low-confidence
       false detection with a robot's ID is averaged into the real one (`_avg_robots` sums
       confidence into a variable it never uses).
     - *Noise settings:* 1 cm and 5 deg match rsim's noise generator, not the real cameras.
     - *First step once logs exist:* replay a recorded vision log through `PositionRefiner`
       and compare velocity jitter and lag against a constant-velocity filter, before
       changing anything.

12. **Sim realism (plan, 2026-10-14):** [`sim_realism.md`](sim_realism.md). Phase 0 (the
    scorecard and the analysis in-repo) and Phase 1 (league ranges from the 2025-26 official
    logs) are done; Phase 2's knobs are built — `SimNoise` seeds latency, kick speed and
    direction spread, two-phase ball friction, per-robot acceleration limits and vision
    noise/dropouts per match, and is wired through `run_match`, `evaluate.py` and `pool_v1`
    (the v3 ranges; the 2026-10-17 round-robin played them as `v3_seed2`). A sensitivity sweep
    was run (results in `sim_realism.md`). Still open: the
    scorecard's remaining gaps (sim keepers save about twice as many good chances, and sim
    shots almost never go wide), the 2022-24 held-out check, choosing which knobs become the
    evaluation default, and using that noise in a round-robin by default (`round_robin.py`
    still needs an explicit `--sim-noise`). P1, the real-data value model, is fitted (passes 1-3,
    2026-10-16/17, in `~/dev/ssl/xg-analysis/value/`; AUC about 0.8 on held-out years) but does
    not yet rank strategies better than goals, so it explains results rather than saving matches
    (`sim_realism.md`, "P1"). P2 (structured strategy search scored by it) is not started.

13. **In-house vision for lab days (team decision 2026-10-25):** [`in_house_vision.md`](in_house_vision.md).
    The 2026-10-22 test day lost most of its time to ssl-vision: cameras 5–8.5 cm apart and 3–9 cm off known spots, colour
    calibration slow and noisy at the edges, fake balls, no way to lock focus or zoom. The plan:
    per-session tag calibration of both cameras, tags on robots, an orange ball inside the field
    mask, sent as standard SSL packets so our stack is unchanged. Lab only; competitions keep
    the league's ssl-vision. The team chose to build it (2026-10-25). Next: a vision scorecard on the
    recorded logs (the ssl-vision baseline to beat), an offline tag calibration solver, then
    recording camera frames on the next test day; nothing is built yet.

14. **Deferred, revisit only when forced** (minimalism):
    - `@pytest.mark.engine` marker for kernel tests — wait for real CI bottleneck data.
    - `TickContext` responsibilities beyond `motion_controller`/`match_log`.
