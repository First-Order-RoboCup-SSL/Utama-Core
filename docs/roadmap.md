# Roadmap / TODO

Running list of larger, not-yet-scheduled workstreams. Unlike
`tactic_model_design_decisions.md` (a decision log for the tactic-kernel specifically),
this is just a place to park bigger ideas so they don't live only in someone's head or
in chat history. Entries get promoted out of here into an actual plan/PR when someone
picks them up — this file isn't itself a design doc.

Resolved work is kept here only as a one-line pointer (what + commit hash) —
the full investigation narrative for anything already fixed lives in git log
(`git log --all --grep=<topic>`) and commit messages, not in this file.

## Done (one-line pointers — see git log for detail)

- **More tactics (first pass)** — `press_and_contain`, `give_and_go`, 5 example `Strategy` configs — `dd73bbc`.
- **BT/py_trees removal** — `AbstractStrategy` rewritten kernel-native, BT scaffolding deleted — `087ee4b`, `960662c`, `48affd6`.
- **CI** — already existed; fixed 2 pre-existing test bugs blocking a green run — `6b0ee7f`.
- **Tournament scoreless-draw debugging** — rSim kick-direction fix, `FastPathPlanner` bugs, `SwitchOfPlayTactic`, `DefenseTactic` foul-loop fix, `TwoDPID` braking cap — `27fd36f`, `e11094a`, `78fce73`.
- **Motion-controller discontinuity handling** — `AbstractPID.calculate()` auto-resets PID state on orientation target jump — `91100ff`, `87cf250`, `036077c`.
- **CustomReferee gaps** (double-touch rule, ball-speed rule, full-episode `reset()`, `set_bt_data` → `set_debug_status` rename) — done; see `docs/custom_referee.md`'s "Known gaps" section (no single commit).
- **Repo root cleanup** — stray session transcripts removed; 7 files importing deleted `strategy.examples` triaged — `74458f7`.
- **Strategy-computation perf pass** — `FastPathPlanner` bounding-box prune, scalar rewrites, `robosim` pipe I/O de-dup, ~3.0x end-to-end speedup — `3086337`, `0cf1e17`, `679e8cd`, `b79b863`, `b98cd29`, `4493002`.
- **`AGENTS.md`** — agent-agnostic contributor doc (no single commit; see file history).
- **Goalkeeper overshoot** — two root causes fixed: goal-line target snap + carrot-based braking cap (`a4df59e`), then a live oscillation traced through the full sim loop and fixed by giving the keeper its own dedicated `PIDController` instead of `FastPathPlanner` — `ba59c9e`.
- **Stuck-match root causes** — `PressAndContainTactic` now drives straight at a fully loose ball; `GiveAndGoTactic` pass-hop handshake gained a timeout (`_MAX_HOP_TICKS`) — `d6b3ff1`.
- **Defense-area retrieval stall** — `_enemy_defense_area_retrieval_exempt()` lets a robot legally retrieve a ball resting in the opponent's defense area — `7a8e717`.
- **Dashboard rebuild** — `custom_referee/gui.py` replaced with a unified Live/Replay/Tournament dashboard, sparse tactic/referee event logging — `c3c389e`.
- **Touchline avoidance + ball-placement-into-defense-area stall** — routing exemption for static obstacles extended from target sanitization to path routing — `3513c59`.
- **SSL rulebook §8.3/8.4 audit + 7 new referee rules** — Pushing, Crashing, Keeper Held Ball, Excessive Dribbling, Robot Stop Speed, Ball Placement Interference, stoppage-time Robot-Too-Close, plus foul-counter/yellow-card mechanism — `434ab29`.
- **Referee restart-formation fixes + strategy override hook** — kickoff steps now read the real goalkeeper ID off the referee packet instead of assuming id 0; new `referee_overrides` hook lets strategies override any restart formation — `7cd1f61`.
- **`robosim` native stdout polluting the JSON protocol pipe** — native C++ layer's stray stdout writes were corrupting the JSON reply pipe, occasionally deadlocking the subprocess; fixed by duplicating the fd before the native extension loads and routing JSON writes through the duplicate — `f613411`.
- **`score_pass_setup` scored almost no real passes** — passer/receiver self-distance dead code in the clearance guard, plus a missing progress term so short low-risk passes always outscored long goal-advancing ones — `dd14f79`.
- **"Permanent ball-lock" in `GiveAndGoTactic`** — `first_touch_stuck` safety valve never released because `hop_count` never legitimately left 0; added a longer `_FIRST_TOUCH_FORCE_SHOT_TICKS` timeout that overrides it — `dd14f79`.
- **`trajsample` planner: robot already inside another obstacle's clearance envelope couldn't plan an escape** — `first_collision_numba` now grants a one-time "still escaping" grace to an obstacle already being penetrated at `t == start_t` — `2e53f3e`.
- **`trajsample` planner: DIRECT_FREE restart target-jitter stall** — `_try_reuse`'s exact-tuple target comparison treated the ball's sub-mm sim jitter as a changed target, forcing a full replan every tick forever; replaced with a small tolerance (`_TRAJECTORY_TARGET_TOLERANCE`) — `5183ed1`.
- **`custom_referee` missing-`designated_position` audit** — `DefenseAreaRule`'s restart-churn bug (`16e26af`) was one instance of a broader gap; audited every rule and fixed 5 more (`KeeperHeldBallRule`, `ExcessiveDribblingRule`, `PushingRule`, `BallSpeedRule`, `DoubleTouchRule`), extracted the shared `RefereeGeometry.legal_restart_position` helper — `2a8c03f`.

## Open

1. **A shared `Sticky`/hysteresis helper** — 5-6 independently-invented
   instances of "keep the previous choice unless a new candidate beats it by
   a margin" exist across skill/tactic/scheduler layers (`go_to_ball.py`'s
   `_COMMIT_RANGE`, `switch_of_play.py`'s `_WEAK_SIDE_MARGIN`,
   `pass_and_shoot.py`'s `_REASSIGN_MARGIN_M`, etc.). Each is well-reasoned in
   isolation but shares no common primitive. **Explicitly not recommended to
   build yet** per the project's minimalism preference — the instances aren't
   quite the same shape (scalar-distance vs. gap-membership vs. boolean edge
   trigger). Revisit only if/when a clearly-6th instance of the exact same
   shape shows up.

2. **More tactics — football-inspired plays/formations.** Only a handful of
   tactics exist today (goalkeeper, defense, pass_and_shoot, lead_and_support,
   switch_of_play, ...). Real football/SSL tactical vocabulary (formations,
   set plays, pressing schemes, overlap patterns) is worth mining for
   genuinely new tactics rather than variations on what exists — also the
   natural forcing function for exercising `TacticTag`/`applicable()` at more
   than toy scale.

3. **AbstractStrategy follow-ups**, deferred from the BT-removal rewrite, not
   urgent. `KernelContext` → `TickContext` rename and the `goalkeeper_id`
   override path are both done (`61a2576`, `7cd1f61`). Still open: whether
   `TickContext` needs new responsibilities beyond `motion_controller`/
   `match_log`.

4. **Idea: geometric intention data for Replay-tab overlays** (user's idea,
   2026-08-24). The intention log currently surfaces *what* changed (which
   tactic a robot holds) but not the *geometry* of a decision — which enemy a
   marker covers, a pass-and-shoot's intended receive point, etc. Looks
   tractable, not speculative: `ShadowAndMarkTactic._assign_marks()`
   (`tactics/shadow_and_mark.py:94-108`) already computes exactly this kind of
   `{marker_id: opponent_id}` mapping every tick and discards it. Natural fit
   for `MatchLog.trace_if_changed()` (tactic-side half is small/mechanical).
   The canvas-overlay half (`dashboard/static/field_canvas.js` has no line/
   arrow primitive yet) is the real design work — not started.

5. **Developer documentation** for a new contributor: how kernel/`Strategy`/
   `Tactic`/`Partitioner` fit together, how to author a new `Tactic` end to
   end, testing conventions (headless rsim for most things, grsim for
   anything dribble-related — see [[project_rsim_dribble_issues]]).

6. **CI: dedicated marker for kernel-specific tests?** `tests/engine/` and
   `tests/strategy_runner/` are the real tactic-kernel surface (no test files
   elsewhere are kernel-specific) — an `@pytest.mark.engine` marker could run
   them fast/prioritized on every push instead of only via the full-suite
   sweep. No pytest markers of any kind exist yet. Flagged for whenever CI's
   actual bottleneck becomes clear from real run times, not done speculatively.

7. **grsim as a CI/tournament environment, alongside rsim** (raised by user).
   - **rsim ball-stickiness bug** — fixed and shipped, `docs/patches/rSim-dribbler-release.diff` — `147d241`. The one test regression this surfaced
     (`test_their_kickoff_clears_our_robots_outside_center_circle`) was a
     test-assertion bug (measured from a fixed field-center point instead of
     live ball position), not a physics/planner bug — also fixed.
   - **grsim headless/dependency/speed investigation** — not yet verified:
     what grsim's actual runtime dependencies are and whether they're
     installable in a GitHub Actions runner at all (grsim is an external
     process — nothing in this codebase starts/manages one today); whether
     grsim genuinely cannot exceed real-time even headless, or whether
     that's a fixable bottleneck the way rsim's was.
   - **CI integration**, blocked on both items above. `.github/workflows/
     ci.yml` currently hardcodes `--ignore-glob "**/*grsim*"`.

8. **Agentic coding infra** (exploratory, no design decided):
   - CI/testing infra shaped for agent iteration loops — fast feedback on
     whether a newly authored `Tactic` is well-formed before a full rsim/grsim
     run.
   - grsim/rsim feedback surfaced back to an LLM in a usable form — today
     results are numbers/logs/plots meant for a human; agents authoring
     tactics need some translation layer (match summaries, failure
     characterizations, maybe rendered trajectory snapshots).

9. **Repo root cleanup — still pending.** 9 `demo_*.py` scripts plus `main.py`
   sit loose in the repo root, no `demos/`/`scripts/` directory. Worth
   deciding whether the survivors (post broken-demo triage) move into a
   proper subdirectory.

10. **Known open bug: FastPathPlanning convergence stall
    (`test_mirror_swap`).** `tests/motion_planning/multiple_robots_test.py::
    test_mirror_swap` is `xfail(strict=False)`. In one specific 6v6 mirrored
    geometry, the two outer "wing" robots (starting at `(-3.5, ±0.75)`)
    consistently stall 0.53-0.54m from their target — inside the episode
    timeout but outside `endpoint_tolerance=0.3`. Reproduced identically via
    plain `move()` commands independent of strategy class, so this is a
    genuine `FastPathPlanning` convergence/local-minimum behavior for this
    geometry, not test flakiness. Not root-caused — a planner-level gap, not
    a one-line patch.

    `DWAController` (already-built, pluggable via `control_scheme="dwa"`)
    resolves this exact scenario outright: 12/12 robots reached, no
    collision, reproduced byte-identical on a second run, vs. fpp's 0/12
    with a collision (`9a84551`). Consistent with the mechanism behind every
    documented `FastPathPlanner` failure in this doc being a subgoal/carrot
    artifact that DWA's per-tick full-resample sidesteps. Not yet known:
    whether `dwa` handles every scenario `fpp` currently handles fine (repro
    tool: `mirror_swap_dwa_probe.py`, parametrized by `control_scheme`).
    Switching the default control scheme is a bigger decision than a single
    bug fix and deliberately left open — see item 13's benchmark findings
    below, which show DWA is not a free upgrade (real per-tick compute cost).

11. **Gameplay bugs observed via dashboard Live view** (flagged 2026-08-24).
    Three of the original four are resolved (goalkeeper overshoot, direct-
    free-kick retrieval, touchline/defense-area placement stalls — see
    "Done" above). One remains open:
    - **Ball-contest deadlock** — believed resolved via `PushingRule`
      (SSL rulebook §8.4.1, symmetric-force no-fault case), confirmed with a
      targeted regression test (`tests/custom_referee/test_ball_contest_deadlock.py`,
      `a3f3795`), deliberately *not* fixed at the `go_to_ball`/planner level.
      Root cause: `FastPathPlanner._path_to`'s ball-adjacent-obstacle
      exemption only applies to target sanitization, not `check_segment`'s
      routing, so the last approach segment to a contested ball is never
      collision-free and the planner detours forever. Extending the
      exemption to routing was tried and reverted — it exposes a worse
      failure (both robots' dribblers register `has_ball` and grind in
      place); this is treated as a genuine 50/50-contest referee-level
      no-fault state instead. (Note: the touchline/defense-area routing fix
      above deliberately does NOT apply here — scoped to static, non-robot
      obstacles only, for exactly this reason.)

    All open items need an actual match trace (via the dashboard's Replay
    tab, or `debug_match.py` + a temporary `trace()`/print hook) before
    attempting a fix — root-cause from real per-tick state, not from the
    symptom description alone.

12. **Testing-gap follow-ups from the §8.4 referee-rules audit** — see
    `docs/testing_gaps.md` for full detail. (1), (2), (3), (5), and the
    Pushing part of (6) closed — `de5e69b`, `f7e9a2a`. **On hold, revisit
    later**: (4) mypy/pyright adoption — would have caught the original
    signature-drift bug for free, but needs its own investigation into how
    much of the existing codebase would fail a cold run. Still open:
    (6, partial) `keeper_held_ball`/`ball_placement_interference` field-
    validation — `keeper_held_ball` fired correctly in live tournament play
    and is closed; `ball_placement_interference` still hasn't fired in any
    live run (reachable and correctly implemented, but current tactics don't
    linger in the placement stadium long enough to trip the 2s grace period
    — not treated as a bug, see `docs/testing_gaps.md` gap #6).

    Two restart-safety bugs found in a follow-up audit before that field
    validation — `docs/testing_gaps.md` gaps #7 and #8, both fixed —
    `ccb172b`.

13. **Trajectory-sampling planner (`trajsampling/`) — architecture-level
    follow-ups, from a TIGERs Mannheim 2024 comparison.** This session added
    `trajsampling` (`5e8844b`), a Numba speedup pass, a per-tick shared-
    obstacle cache, a `BangBang1D` endpoint-continuity fix (`7370736`), and
    the first dedicated test coverage plus a scheme-agnostic standardized
    benchmark harness (`68055d9`, `docs/motion_planning_comparison.md`) that
    `trajsampling`/`fastpathplanning`/`dwa` had no common comparison for
    before. All items below assume that harness as the way to validate any
    future change here.

    An external research-agent review compared this implementation against
    the TIGERs Mannheim 2024 champion paper's published trajectory-sampling
    design (the architecture this module is modeled on) and flagged the
    still-open gaps below, in roughly the order worth tackling them:

    - **Directional-tube enemy-obstacle model (attempted, reverted — start
      here).** `EnemyRobotObstacle` (`obstacles.py`) currently models a
      moving enemy's reachable region as an isotropic expanding circle.
      TIGERs' paper uses a directional tube instead — reachable envelope
      grows mostly along the enemy's current heading — specifically to
      reduce unnecessary detours around opponents the circle over-avoids.
      A capsule-shaped implementation was attempted and reverted: it
      regressed `mirror_swap` from "0 collisions" to a reproducible 1.1mm
      collision, and rsim sensor jitter on a "stationary" enemy picked a
      near-random tube direction, doubling `static_slalom`'s completion
      time even after a stationary-speed-threshold fix. **Next attempt
      should**: size the lateral bound to stay strictly conservative
      relative to the old circle at small time horizons (e.g.
      `lateral = max(0.5*a_max*tc**2, some old-circle-derived floor at that
      tc)`), and debug `static_slalom`'s regression and `mirror_swap`'s
      collision as two separate problems in separate passes, using the
      benchmark suite's per-scenario cells to isolate each before combining.
    - **`Trajectory2D` discards lateral (transverse) velocity.** It's a
      2D-space wrapper around a single `BangBang1D` solve along the straight
      line from `p0` to `p1` (`bang_bang.py`) — any velocity component
      transverse to that line is silently dropped at trajectory start,
      which can demand an instantaneous direction change, most risky right
      after avoiding another robot or switching targets. Frequent
      replanning masks this in practice but doesn't enforce acceleration
      limits at the discontinuity itself. Architecture-level change to the
      hot path both `BangBang1D`/`Trajectory2D` and the Numba kernels
      assume the current single-axis shape — wants a concrete failure case
      (a replay showing a bad velocity-direction snap) before starting, and
      a full before/after benchmark run given the blast radius.
    - **Every trajectory ends at zero terminal velocity.** Forces
      accelerate-brake-to-zero-accelerate cycles for continuous motion
      (interception, support repositioning, dribbling) — TIGERs identify
      this as a known limitation of their own earlier system. Likely the
      single largest behavioral improvement available here per the external
      review, but coupled to the lateral-velocity item above (do that item
      first).
    - **Fixed robot-ID priority ordering is a valid symmetry-breaker but
      strategically weak.** `robot_id > other_id` prevents two robots from
      both deciding the other yields, but can make the wrong robot yield. A
      stable multi-key ordering (goalkeeper/restart safety > possession/
      interception urgency > tactic role > distance-to-target > robot ID as
      final tie-break) would live in the planner's priority policy — really
      a `[[project_tactic_model]]`-adjacent concern more than a pure
      motion-planning patch.
    - **No explicit stop/yield planner result.** When every candidate
      collides, the planner currently returns the longest-surviving
      candidate and relies on the controller's residual emergency brake.
      Making "yield" a first-class planner return value would apply when
      the first collision is imminent, every candidate is priority-blocked,
      or the robot has no valid route.
    - **Collision sampling is adaptive-timestep, not a formal swept/
      continuous check.** Can still in principle miss a narrow collision
      event between samples for a fast-moving obstacle. Bounding timestep by
      relative speed in addition to distance, or an analytic time-of-impact
      calculation, would close this. The external review rates this as
      higher-value than further raw speed optimization — a correctness-
      margin question, not a performance one.
    - **Intermediate-target sampling is random, not geometry-guided.**
      Matches the TIGERs paper's own approach (not wrong), but can produce
      poor samples in narrow corridors and makes rare failures hard to
      reproduce. Adding a handful of deterministic candidates (tangents
      around the first blocking obstacle, obstacle-normal offsets, the
      previous winner) alongside the existing random samples, ranked by
      progress/clearance/duration/continuity, would improve reproducibility
      without giving up random exploration.
    - **Stale committed-trajectory reuse could validate more.** Current
      reuse check: same target (within `_TRAJECTORY_TARGET_TOLERANCE`),
      robot still near predicted position, trajectory still collision-free,
      priority checks still valid. Possible additions: compare actual vs.
      planned velocity, invalidate on a sharp obstacle-velocity change,
      invalidate when a new obstacle enters the swept corridor, cap maximum
      reuse age.
    - **Bigger-picture, not yet decided**: DWA independently resolves at
      least one known `FastPathPlanner` local-minimum failure (item 10
      above). A DWA-as-short-horizon-recovery-mode hybrid is plausible but
      is a planner-selection architecture question — don't start it
      speculatively unless a concrete case emerges from further findings.
    - Only after the above: consider Ruckig or another jerk-limited multi-
      axis trajectory generator as a full primitive replacement — the
      nearer-term weaknesses are in obstacle modeling and multi-robot
      coordination, not the bang-bang algebra itself, so this is explicitly
      a last item.

    **Locked baseline benchmark (git rev `7370736`, `68055d9`).** Full
    12-scenario × 3-scheme matrix, confirmed reproducible (byte-identical
    across 5 repeats — rsim is deterministic). Pass rate: fpp 9/12, dwa
    8/12, trajsample 10/12 — **no scheme sweeps the board**, each has
    distinct real failure modes: `mirror_swap` (dense 6v6) — fpp collides,
    dwa passes cleanly, **trajsample stalls at `sim_timeout`** (8/12
    reached, 0 collisions, thrashing metrics ~5x the next-highest cell) —
    new, still-open, may share a mechanism with the reverted directional-
    tube regression above, check before treating as separate fixes.
    `crossing` (2 robots, perpendicular) — fpp and dwa both collide, only
    trajsample passes. `narrow_passage` (0.24m gap) — fpp and trajsample
    pass, dwa collides (matches the strict `xfail` in `test_scenarios.py`).
    `static_slalom`/`grid_intersection` — dwa fails both, fpp and trajsample
    pass both. **DWA's per-tick compute cost is consistently the highest of
    the three** (2-3x trajsample's in `mirror_swap`/`crossing`), matching
    the user's recollection of DWA lagging in live matches — any future
    default-scheme decision (item 10) should weigh this, not just pass/fail.

    Three real bugs were found and fixed while getting `trajsample` to a
    playable state (`c4ad99c`/`2e53f3e`/`fe6a07e`/`5183ed1`, see "Done"
    above for the last two): a `Vector2D`/tuple type mismatch in
    `block_shape.py` that only `TrajectorySamplingController` was strict
    enough to crash on; the ball treated as an unconditional collision
    obstacle for the robot fetching it (fixed via a per-`plan()`-call ball
    exemption); and `Trajectory2D.compute`'s degenerate zero-distance
    fallback dropping all lateral velocity when planning a stationary
    orient-in-place target.

    **Still open, found while verifying the ball-obstacle fix above — not a
    trajsample-specific bug, see `docs/testing_gaps.md`'s same-team-scrum
    finding.** A `GiveAndGoTactic` carrier and a `DecoyOverloadTactic` decoy
    could independently converge on the same possessed ball with no
    cross-tactic awareness; root-caused and fixed via
    `_teammate_already_has_ball()` — `a59a8e5`. This is the same mechanism
    class as `docs/testing_gaps.md`'s cross-tactic ball-collision freeze
    (picker-level, `0e510a0`/`ae6a6f3`) — a recurring architecture gap
    (nothing checks "is another already-assigned robot also converging on
    this exact ball") worth a general fix if a third instance shows up.

14. **Outer-loop strategy evaluation (inner loop / outer loop split).** The
   goal is coding agents iterating strategies against the evals with minimal
   human replay-watching. Today the 65 s `tournament.py` round-robin is a
   stall fuzzer, not a strategy evaluator: nearly every match is a scoreless
   draw, rsim is deterministic so re-running a matchup adds no information,
   and a planner stall is indistinguishable from a bad strategy in the score.
   Win-rate/Elo stays the objective, but it is too sparse and too expensive to
   be the only signal. Design, in three tiers with three homes:

   - **Inner loop = contract tier (test suite, binary, blocks merge).** Tactic
     contracts, kernel invariants, and zero `StallEvent`s across N seeded
     matches with `RestartFuzzingReferee` on. No metrics live here: a metric
     has no pass threshold that stays true as strategies improve. Built so
     far: stall watchdog + `--strict`, commitment deadline, restart fuzzer,
     pure `Game` builder, regression tests for every recent fix. Missing:
     `--fuzz-restarts SEED` wiring in `tournament.py`, and CI running the
     strict seeded gate.
   - **Outer loop, fast half = scenario bench (a benchmark like
     `tools/motion_planning_benchmark.py`, numbers vs a committed baseline,
     non-blocking).** A scenario is a seeded start state plus a 15-30 s
     horizon (kickoff, direct free near the box, loose ball at midfield, 3v2
     counter, defending a corner). `scenario_from_replay` is the harvester:
     every restart in every replay is a candidate, so the bank grows for free
     and can be weighted toward situations the last ladder run lost. Scored
     by *calibrated proxy metrics* (below), always as differentials vs the
     opponent, never absolute.
   - **Outer loop, slow half = ladder (tournament, acceptance gate for
     "promote to best").** Not round-robin: the candidate plays a frozen
     reference pool of 4-5 strategies spanning naive to current best, both
     colours, K seeds each; Elo anchored to the pool so it cannot drift. A
     promoted candidate joins the pool and the most redundant member leaves.
     Full 600 s matches only against the top of the pool, for final
     acceptance.

   **Metric design: derive, don't invent.** Add cheap event counters to
   `MatchStats` (shots / on target, attacking-third entries, completed
   passes, turnovers, possession-under-pressure seconds, restart-to-first-shot
   time, each with the opponent counterpart), then regress goal difference on
   the differentials over existing full-match data and keep only the ones
   with predictive weight. The weighted sum is the proxy score, calibrated in
   goals. Goodhart guard: whenever the bench improves but the ladder does
   not, the proxy is being gamed — retire or reweight it. That is the only
   place the expensive signal is spent on metric design. First step (in
   progress 2026-09-03): offline correlation study over the three 231-match
   65 s runs in `replays/` (`tools/metric_correlation.py`) to see which
   proxies have any per-match or per-strategy signal before instrumenting.

   **Compute discipline (determinism is an asset here):** paired comparison
   with common seeds (candidate and baseline play identical seeds and
   opponents, so the variance of the *difference* is small); sequential
   stopping (run seeds in batches, stop once the paired difference is clearly
   positive/negative/nothing); short horizons from sampled states rather
   than long matches (a 600 s match yields ~a dozen independent situations,
   thirty 20 s scenarios yield thirty for a tenth of the compute); one
   `evaluate <strategy> --budget` entry point that runs contracts + bench at
   low budget and adds the ladder at high budget, and writes
   `docs/strategies.md` itself. Consolidating `tournament.py` /
   `full_match_tournament.py` / `arena_tournament.py` is a prerequisite.

   **Build order:** event counters in `MatchStats` → one calibration
   tournament (full-match, competitive tier) → scenario bench with harvested
   states → reference ladder with paired seeds. The first two are ~a day and
   tell you whether the proxies carry any signal before the rest is built.
   Sim-fidelity caveat: rsim's dribble physics are known-flaky, so keep a
   short list of behaviours rsim is not trusted for and spot-check anything
   on it in grsim before believing a bench or ladder gain that depends on it.

   **First data point (2026-09-03, `tools/metric_correlation.py` over the
   three 231-match 65 s trajsample runs):** turnovers, completed passes and
   attacking-third entries are both valid (rho 0.65-0.73 vs points per
   strategy) and reliable run-to-run (rho 0.79-0.86); shots are valid but
   unreliable at 65 s; possession and robot-motion carry no signal. Caveat
   that applies to *all* of that data: see item 15.

   **Opponent counterparts + possession-under-pressure + restart-to-entry
   ported live (2026-09-04):** closed the "each with the opponent
   counterpart" gap above. `MatchStats` previously only had friendly-side
   `turnovers`/`completed_passes`/`attacking_third_entries` even though
   `tools/metric_correlation.py`'s offline definitions were always computed
   both-sided (`completed_passes_friendly`/`_enemy`, etc.) — the live
   accumulator was the one that had fallen behind its own offline
   counterpart, not a deliberate scope decision. Added
   `enemy_turnovers`/`enemy_completed_passes`/`enemy_attacking_third_entries`
   (same possession-radius state machine, now attributing both sides
   instead of only friendly), `possession_under_pressure_s` (metric 4:
   seconds a side's nearest-to-ball robot also has an opponent within
   `_PRESSURE_RADIUS_M`, accumulated as real elapsed sim-seconds via
   `game_frame.ts` deltas rather than assuming a fixed tick rate, since
   `record_tick` doesn't see rsim's configured step rate), and
   `restart_to_first_entry_s`/`n_restarts`/`n_restarts_with_entry` (metric
   5: a live-play command's start clock, attributed to whichever side is
   nearest the ball at that instant, stopped at that side's own
   attacking-third entry — new `_maybe_start_restart_clock`, called before
   `_maybe_record_stalls` overwrites the restart-command-transition state
   both watchdogs share). All three match the offline tool's definitions
   exactly (same thresholds/state machines), not new proxy designs of
   their own. 8 new tests, 1 existing test's assertion corrected (an
   enemy-to-enemy handoff was asserted to tally nothing; it now correctly
   asserts `enemy_completed_passes == 1`). Full suite green (4193 passed).
   `restart_to_first_shot` (the docstring's other named metric) was not
   added — no offline equivalent exists in `tools/metric_correlation.py` to
   port from (metric 5 there is entry-based, not shot-based); would need
   its own design pass rather than a direct port.

   **Scenario bench design + v1 schema slice (2026-09-04).** Before touching
   `replays/` again: this session's own "clean" post-fix replay runs hadn't
   been confirmed clean, and a 925-file "complete" run turned out to be
   entirely pre-fix data stuck at kickoff — harvesting from `replays/`
   without a trust gate would have poisoned the bank on day one. Design
   settled instead of assumed:
   - **Source, not byproduct.** Scenarios come from three tagged places, not
     an arbitrary `replays/` scrape: calibration/ladder matches at the
     current evaluator version (main source, pool-vs-pool not just champion
     games), lost plays from the last ladder run (a separately tagged
     *weakness* subset, not the whole bank), and hand-authored canonical
     anchors that never move. Match-level gate: tagged run, current
     evaluator version, `stall_events == 0`. Pre-fix replays are discarded,
     not filtered.
   - **A restart is the transition into live play**, not PREPARE (low
     information, positioning test with a known answer):
     `PREPARE_KICKOFF_*/PREPARE_PENALTY_* → NORMAL_START`,
     `STOP → DIRECT_FREE_*`, `STOP → FORCE_START`. `BALL_PLACEMENT_*` is its
     own small family. Open play is a second, `MatchStats`-event-triggered
     harvest mode (possession change, attacking-third entry, loose ball,
     numerical-advantage detector), scored with a wider noise floor and
     lower composite weight since it can't get the restart family's free
     re-partition. Every restart yields two scenarios (candidate kicking,
     candidate defending) — restarts are asymmetric.
   - **Mem-loss fix is a lead-in, not serialized tactic state** — the bank
     must stay policy-agnostic (a candidate that restructures its tactic
     layer can't consume the champion's serialized `mem`). Event-triggered
     scenarios start 1-2s before the anchor tick so both policies get a
     runway to reconstruct roles; this doubles as a robustness test (can't
     recover roles in a second = brittle, same as a vision dropout).
   - **Validity is static + dynamic.** Static: in-bounds, no overlap,
     physically plausible speeds, both teams present — implemented now (see
     below). Dynamic (deferred, needs real match data): play forward
     champion-vs-champion and vs pool across seeds; classify dead (no
     discriminative power, drop) / determined (near-zero information, drop
     or keep one anchor) / noisy (keep only if the family's noise floor
     absorbs it) / informative (keep). Plus a human spot-check, five per
     family, at bank creation.
   - **Bank v1 target shape** (not yet built): ~20 hand-authored anchors +
     ~100 restart-triggered (both perspectives, frequency-weighted) + ~30
     lost-play weakness + ~30 event-triggered open-play (lower-trust) ≈ 200
     total. Immutable per bank version — adding scenarios makes a new bank
     ID and forces a champion re-baseline. Lifecycle per scenario:
     candidate → validated → active → retired, only `active` scores.
     Provenance (source run, evaluator version, anchor tick, trigger,
     family) lets a contaminated batch be retired by query. Gating plan:
     warn-only for two weeks against real candidates before restart
     families gate merges; open-play families stay advisory until their
     ladder correlation is measured. Rationale: a wrongly-excluded scenario
     costs a little coverage; a wrongly-included one teaches an optimizer to
     game a bug — bank should err small and clean.

   **Built in two passes (2026-09-04): schema/anchors first, then
   harvester/screen/scorer/CLI — everything except an actual fresh
   calibration run.**

   *Pass 1 — schema + hand-authored anchors* (first slice, chosen because
   it has no replay/tournament dependency): `utama_core/replay/
   bench_scenario.py` (`BenchScenario`/`ScenarioProvenance`/
   `ScenarioTrigger`/`ScenarioFamily`/`ScenarioLifecycle` + `static_screen()`,
   wrapping `scenario.Scenario` rather than replacing it, so hand-authored
   and harvested scenarios share one downstream `apply_scenario` path) and
   `utama_core/replay/hand_authored_scenarios.py` (4 anchors —
   `kickoff_center_v1`, `direct_free_defending_near_box_v1`,
   `direct_free_attacking_near_box_v1`, `open_play_3v2_counter_v1`).

   *Pass 2 — the rest of the pipeline, minus the actual tournament run*:
   - `utama_core/replay/scenario_harvester.py`: `find_restart_transitions`
     scans a `.intentions.jsonl` sidecar for the exact transitions the
     design calls for (`PREPARE_KICKOFF_*/PREPARE_PENALTY_* →
     NORMAL_START`, `STOP → DIRECT_FREE_*`, `STOP → FORCE_START`, deduped
     within 0.5s); `match_is_trustworthy` enforces the harvest gate
     (`<match>.stats.json` must exist and report `stall_events == []`,
     fails closed on anything missing/unparseable — the exact gate that
     would have caught the 925-file pre-fix contaminated run this session
     found); `harvest_run_dir` walks a completed `tournament.py` run
     directory end to end and returns scenarios plus a
     matches-seen/trusted/untrusted report. Not yet exercised against a
     real tournament run — tested against synthetic `.npz` replays
     (`ColumnarReplayWriter`) + hand-written sidecars/stats files instead.
   - `utama_core/replay/dynamic_screen.py`: plays a scenario forward
     champion-vs-self plus a small pool, classifies DEAD / DETERMINED /
     NOISY / INFORMATIVE per the design's outcome-variance rule. Pool is
     policy variation, not RNG seeds — rsim is deterministic given
     identical inputs, so "several seeds" here means several opponent
     pairings (see `tools/metric_correlation.py`'s same observation).
   - `utama_core/replay/scenario_scorer.py`: `score_scenario` builds a
     fresh headless runner (same construction `repro_from_replay.py`
     uses), applies the scenario (ticking `lead_in_s` first for
     event-triggered scenarios — the mem-loss lead-in fix from the design
     pass, not yet exercised since nothing sets `lead_in_s` nonzero yet),
     ticks `horizon_s`, and classifies a `ScenarioOutcome` (GOAL_AGAINST
     … NEUTRAL … GOAL_FOR, ordinal) plus a foul flag from `MatchStats`
     deltas and direct goal-line geometry checks (`RefereeGeometry`) —
     reuses existing counters, invents no new metric.
   - `tools/scenario_bench.py`: CLI mirroring `motion_planning_benchmark.py`'s
     shape. `--list-scenarios` prints the loaded bank; `--dynamic-screen`
     runs the screen and reports verdict counts; the default mode runs
     PAIRED scoring (candidate vs opponent, baseline vs same opponent, same
     scenario) and reports per-scenario and per-family delta as JSON +
     Markdown. `--harvest-from RUN_DIR` adds harvested scenarios to the
     hand-authored bank; `--families` filters. Smoke-tested end to end
     (list, paired score, dynamic screen) against
     `build_default_kernel_strategy` — all three paths produce sane output.

   26 new tests total across both passes (`test_bench_scenario.py`,
   `test_hand_authored_scenarios.py`, `test_scenario_harvester.py`,
   `test_scenario_scorer.py`, `test_dynamic_screen.py`). Full suite green.

   **Still not done, deliberately out of scope for this pass:** the actual
   fresh calibration tournament run (`harvest_run_dir` is untested against
   real match data — only synthetic fixtures), the lost-play weakness
   subset and event-triggered open-play harvesting (both need ladder-run
   history that doesn't exist post-fix yet), bank versioning/lifecycle
   persistence (`ScenarioLifecycle` exists as a field but nothing
   promotes/retires a scenario or writes a bank manifest), and the ladder
   (slow half) entirely.

15. **`trajsample` liveness floor: stalls down from 106/231 to 27/231, not
    yet zero; the BangBang1D fix cannot land until the planner handles
    blocked starts.** Current position (read this before trusting any count
    below): the last *full* 231-match total was **27/231 stalled** after
    `4b701ae` (RESTART_STALL 9, COMMITTED_FROZEN 16, NO_PROGRESS_POSSESSION
    2). Two families were reduced after that run without a fresh full
    re-run — the 20 `BALL_PLACEMENT_*` corner-overshoot `RESTART_STALL`
    cases were fixed 2026-09-12, and the live-play `COMMITTED_FROZEN`
    ball-hold family was taken to **8 matches / 9 stall events** (`70cb5c6`).
    So there is no single confirmed total newer than 27/231; the residual
    mechanism is documented at the end of this item. Everything below is the
    dated investigation trail that got here, oldest first — the 106/231 and
    127-stalled figures in it are historical baselines, not current state.
    All counts measured with the stall watchdog (`tournament.py` STALLS
    section, `--strict`), same seed, 65 s, 231 matches:

    - True HEAD baseline before `885eba4` (`bff5321`): 127 stalled matches
      (DIRECT_FREE 90, NORMAL_START 17, STOP 11, PREPARE_KICKOFF 6,
      BALL_PLACEMENT 2, FORCE_START 2, PREPARE_PENALTY 1), 12 decisive.
      Three distinct DIRECT_FREE mechanisms were traced in replays: (1) a
      stale committed trajectory of a robot moved outside `plan()` (every
      non-kicker gets `empty_command()` in `DirectFreeOursStep`; the keeper
      always) acting as a ghost obstacle that carries the robot's priority
      and blocks the kicker for the rest of the match - dominant, fixed in
      `885eba4` by gating the committed obstacle on divergence from
      `trajectory.state_at(min(t, duration))`; (2) reuse-tolerance drift of
      the committed target - a refresh-on-reuse fix was tried, guarded and
      unguarded, and made the round-robin *worse* than HEAD (162 stalled,
      127 DIRECT_FREE), so it is not applied; (3) STOP-phase keep-out
      stalls (11 matches) that self-resolve after the 15 s stop timeout.
      Gating (1) on elapsed time alone instead of divergence fires on every
      routine plan completion and froze kickoffs - a robot resting on its
      completed endpoint must keep its priority.
    - After `885eba4`: 106 stalled (DIRECT_FREE 57, NORMAL_START 34,
      FORCE_START 8, PREPARE_PENALTY 6, PREPARE_KICKOFF 5, BALL_PLACEMENT 4),
      17 decisive, goals 12 -> 17, shots 29 -> 45, robot motion 0.33 -> 0.41.
      Live-play `COMMITTED_FROZEN` rose 19 -> 42; every traced case is a
      *tactic* ball hold, not a planner stall: a three_slot defender parked
      with the ball 0.10 m in front of it at 31 s / 43 s regardless of
      opponent, a low_block defender sliding along x = -3 with the ball
      glued to its dribbler, a high_press/press_and_pass freeze at 19.8 s
      after FORCE_START. HEAD shows the same holds; matches now reach them
      instead of dying earlier. Those holds are the next liveness target
      (same family as the GiveAndGo ball-lock fixed in `dd14f79`).
    - `ball_travel_m` is not a play-quality signal: HEAD's median (12.2 m vs
      9.2 m after the fix) is inflated by a robot dribbling in a 0.4 m circle
      at 0.8 m/s for 50 s (low_block vs overload_flow family). Prefer the
      item 14 counters (turnovers, completed passes, attacking-third
      entries) and the STALLS section.
    - **57 DIRECT_FREE stalls, first trace (2026-09-04).** Root-caused one
      real mechanism and shipped a fix, but it only resolved 1/57 in the
      full round-robin (57 -> 56 DIRECT_FREE, 106 -> 105 total) — much
      smaller than expected; see the important caveat below about why
      small-subset re-runs overstated it.
      - **Mechanism found and fixed**: `TrajectorySamplingPlanner.plan()`
        has always taken an `exempt_defense_area` parameter mirroring
        `FastPathPlanner`'s existing opponent-defense-area-retrieval
        exemption (the same class of bug as the "Defense-area retrieval
        stall" fix in Done, `7a8e717` — but that fix only ever touched
        `fpp`). No caller ever passed it, so `trajsample` always treated the
        opponent's box as a hard wall, including during a legitimate
        DIRECT_FREE_OURS/BALL_PLACEMENT_OURS retrieval when the ball itself
        rests inside it — the kicker's own approach target sits inside an
        obstacle it can never enter. Fixed by giving
        `TrajectorySamplingPlanner` its own `_enemy_defense_area_retrieval_exempt()`,
        computed internally from `game` exactly like `FastPathPlanner`
        already does, called from `plan()` whenever the caller doesn't pass
        the parameter explicitly (nothing currently does). Confirmed via a
        debug trace that the exemption fires correctly at runtime
        (`exempt=True` on every `plan()` call for the affected robot/tick).
        Fixed exactly one match in the full round-robin:
        `high_line_zone_vs_high_press`.
      - **Important caveat, discovered the hard way**: a match's stall
        outcome is NOT reliably reproducible by re-running just that one
        pair in a small subset (`tournament.py low_block tiki_taka`, say) —
        several matches that appeared fixed in small 2-4-config smoke tests
        (`clear_danger_vs_clear_press_plus`, `clear_press_plus_vs_counter_press`,
        `three_slot_vs_tiki_taka_plus`, `split_shape_vs_tiki_taka`, and
        others from a first-pass classification that guessed ~27/57 were
        this same defense-area bug) turned out to still stall, identically,
        when re-checked cleanly in isolation after ruling out test
        contamination (a `git stash`/`pop` cycle run concurrently with a
        background full-round-robin process while validating a different
        change — do not edit/stash a file a background tournament run is
        currently importing from worker processes). **Always confirm a
        stall fix against the full 231-match round-robin** (or at least the
        planned 30-40 match liveness subset, once it exists — see below),
        not an ad hoc small-catalog re-run.
      - **Genuinely different second mechanism, seen while re-tracing
        `clear_danger_vs_clear_press_plus`** (the exemption fix does NOT
        help here — confirmed firing correctly but the match still stalls
        identically, onset t=43.8s): the kicker's direct path to the ball
        collides with real, legitimately-positioned enemy robots defending
        near their own goal/corner (e.g. two enemies within ~0.3-0.9m of
        the ball's resting spot at (-4.25, 0.70)), and none of `plan()`'s
        sampled intermediate-target candidates ever find a collision-free
        route either — the kicker sits ~3.6m away the whole restart,
        making essentially no progress. Looks like a genuine multi-robot
        local-minimum/congestion case in a crowded corner, not a rules-
        exemption gap — a harder problem, not traced further this session.
      - **Third mechanism found and fixed (2026-09-04), live-traced (not
        replay-inferred) on `clear_danger_vs_clear_press_plus` itself**:
        `DirectFreeOursStep.update()` (`custom_referee/actions.py`) picked
        the kicker fresh every tick via `min(game.friendly_robots,
        key=distance_to_ball)`, with no hysteresis. Live per-tick trace
        (not replay data — `debug_match.py` with a temporary probe script
        reading robot state directly) showed three robots sitting within
        2cm of each other's distance to the ball; the "closest" identity
        flipped on ordinary rsim position noise **243 times in 27 seconds**
        (~9/s) and never stopped. Every flip reset the newly-chosen kicker
        to a standing start (the old one dropped to `empty_command()`
        immediately), so no robot ever held the role long enough to close
        meaningful distance — this is the exact bug shape `[[Sticky/
        hysteresis]]` (item 1) already names, and the exact fix already
        shipped once for the same shape in `pass_and_shoot.py`'s
        `assign_passer_receiver`. Fixed identically: `DirectFreeOursStep`
        now holds a `shared.tolerance.Sticky[int]` (`_kicker_sticky`,
        `margin=0.3` — same value as `pass_and_shoot._REASSIGN_MARGIN_M`)
        across ticks, since the class is already constructed once per match
        and reused (`RefereeOverride.__init__`, same lifetime pattern
        `BallPlacementOursStep`'s existing `_placer_id` sticky field
        relies on). Confirmed live: kicker switches during the restart
        dropped from 243 to 4 (one legitimate hand-off once robot 3 was
        genuinely closest, then held). Two new regression tests added
        (`test_kicker_choice_is_sticky_across_near_tied_distances`,
        `test_kicker_reassigns_when_a_robot_is_clearly_closer`).
        **Important, does not fully resolve this match or the 56-stall
        backlog**: with thrashing gone, robot 3 now visibly commits and
        closes ground (min distance to ball drops from ~3.9m to ~2.6m by
        t=56) — but then the *second* mechanism above (multi-robot
        congestion/local-minimum) takes over: it oscillates between ~2.6m
        and ~3.4m, retreating and re-approaching, for the rest of the 90s
        window, never reaching the ball. `clear_danger_vs_clear_press_plus`
        and a second spot-checked match (`three_slot_vs_tiki_taka_plus`)
        both still record a `RESTART_STALL` at the identical onset time
        with this fix applied — so per this session's honest-caveat
        precedent (small-subset re-runs previously overstated a fix's
        reach), assume this fixes some unknown subset of the 56 sharing
        pure identity-thrashing with no congestion underneath, not the
        backlog as a whole, until re-classified against a full round-robin.
        The congestion/local-minimum mechanism (previous bullet) is now the
        clean, thrashing-free target for that investigation — the noise
        that made it hard to trace live is gone.
      - **`BallPlacementOursStep` hardened with the same fix (2026-09-04),
        not confirmed as a root cause**: had the identical unguarded
        `min(..., key=distance)` shape on every tick before the ball reaches
        `designated_position` (only the post-arrival release hold used the
        existing `_placer_id` field). Given a `Sticky[int]` `_placer_sticky`
        (`margin=0.3`, same value/rationale as `DirectFreeOursStep`'s
        `_kicker_sticky`), reset alongside `_placer_id` in `_reset_release`.
        Risk window is narrower than `DirectFreeOursStep`'s: non-placer
        robots are actively cleared away every tick via
        `_clear_to_legal_positions`, so a tie self-resolves within a tick or
        two rather than persisting for a whole restart — applied as
        hardening against the known-bad pattern, not confirmed live against
        an actual thrashing trace (unlike the `DirectFreeOursStep` fix,
        which had a live 243->4 trace). Two regression tests added
        (`test_placer_choice_is_sticky_across_near_tied_distances`,
        `test_placer_reassigns_when_a_robot_is_clearly_closer`), mirroring
        the `DirectFreeOursStep` sticky tests. Whether this moves any of the
        4 BALL_PLACEMENT stalls is unverified — check against the next full
        round-robin.
      - **Fourth mechanism found and fixed (2026-09-04):
        `TrajectorySamplingController`'s emergency-brake layer
        (`controllers/trajsampling.py::calculate`) scaled the robot's raw
        current velocity (`robot.v`) instead of the planned trajectory's
        velocity (`vx, vy` from `result.trajectory.state_at(lookahead)`)
        when braking.** Whenever the robot's actual momentum pointed
        anywhere other than where `plan()` said it should go (e.g. right
        after being nudged off-course near a crowded obstacle), a braking
        tick commanded the robot further along its OLD heading instead of
        correcting it toward the target. Live-traced on
        `clear_danger_vs_clear_press_plus`'s DIRECT_FREE stall: braking
        fired on ~35% of ticks near a crowded obstacle, and dozens of
        subsequent replans were each triggered by a ~0.08-0.09m position
        deviation — right at `_TRAJECTORY_POSITION_TOLERANCE` — even though
        `plan()` reported a clean, collision-free, converges-to-target
        trajectory on every single call. Mechanism: brake drifts the robot
        off its committed straight-line path by just enough to invalidate
        it via `_try_reuse`'s position-tolerance check; `Trajectory2D.compute`
        drops transverse velocity at the start of every fresh replan (the
        already-documented item 13 limitation), so the corrective replan
        itself launches the robot on a new heading with no continuity from
        its actual motion — feeding a retreat-and-reapproach loop that
        persisted for the rest of the match. Fixed by scaling `(vx, vy)`
        instead of `robot.v`; verified live on two previously-stalling
        matches (`clear_danger_vs_clear_press_plus` and a second spot-check),
        both now stall-free, plus a new regression test
        (`trajsampling_controller_test.py::test_brake_scales_planned_velocity_not_robot_raw_velocity`)
        that fails against pre-fix code and passes against the fix.
        `git log` confirms this line was never touched since the file's
        original commit (`5e8844b`) — a latent bug, not a regression.
      - **Full 231-match round-robin with the brake-direction fix
        (2026-09-04, `tournament_20260904_075706` vs. baseline
        `tournament_20260903_234518`)**: raw stall count went UP, 106 -> 118
        (61 matches newly fixed, 73 newly stalled — net -12). This is not a
        new bug from the fix: every newly-stalled match checked
        (`clear_danger_vs_overload_flow`) shows the identical
        retreat/re-approach oscillation signature as the mechanism above,
        and the baseline's `summary.json` confirms it did NOT stall before
        (0-0, no `stall_events`). With the brake no longer accidentally
        halting robots early on their own drifted heading, more robots now
        travel their FULL intended path far enough to reach the still-open
        multi-robot congestion/local-minimum mechanism (second bullet,
        above) — the fix removes one bug and, in doing so, exposes more
        matches to the other, already-known one. Despite the higher raw
        stall count, the fix is net-positive on every item-14 quality
        signal: goals 17 -> 21, decisive matches 17 -> 21, shots 45 -> 61,
        completed passes 527 -> 560; turnovers roughly flat (307 -> 295)
        and attacking-third entries slightly down (225 -> 203, consistent
        with more robots getting caught in congestion before completing an
        entry). Ship the fix (it is a correctness fix for a real latent bug
        with no legitimate case where scaling raw velocity was ever right)
        but do not count it as DIRECT_FREE-stall progress — the honest
        framing is that it converts some early, accidental "stalls" into
        either goals or into hitting the *real*, still-open congestion bug
        further down the line.
      - **Fifth mechanism found and fixed (2026-09-04) — this IS the
        multi-robot congestion/local-minimum mechanism from the second
        bullet above, root-caused**: live-traced on
        `clear_danger_vs_shadow_switch`'s DIRECT_FREE_BLUE stall.
        `TrajectorySamplingPlanner._intermediate_targets` always retries the
        previous tick's winning two-segment detour target (`last`) FIRST
        (see its own docstring — this exists to stop tick-to-tick direction
        flipping, a real and separate problem it correctly solves), and
        `plan()` commits to the first collision-free candidate it finds
        without ever comparing it to the freshly-sorted, actually-toward-
        target candidates later in the list (see `plan()`'s early-out at the
        top of its fallback loop). When the direct path is genuinely,
        repeatedly blocked by a real obstacle, `plan()` falls through to
        this method every tick; if `last` happens to be a stale point
        sitting in open space *behind* the robot (chosen once, for some now-
        irrelevant earlier situation), it stays collision-free indefinitely
        and so keeps winning the early-out forever, never re-validated
        against direction — only against "still collision-free." Traced
        exact numbers: `last` was 166 degrees off the current goal
        direction; each replan cycle committed a short first-leg burst
        toward it (backward), then switched after 0.2-0.4s to a second leg
        that had to kill that backward velocity before making any real
        progress — net near-zero displacement, repeating every ~1s for the
        rest of the restart, all while `plan()` reported `has_collision:
        False` on every single call (a stall with a "clean" planner output
        the whole time, which is why static/replay-only classification
        couldn't distinguish it from a genuine local minimum). Fixed in
        `_intermediate_targets` by dropping `last` outright (not merely de-
        prioritizing it) whenever its angular distance from the current
        p0->final_target direction exceeds
        `_STALE_INTERMEDIATE_TARGET_ANGLE_RAD` (90 degrees) — a real sidestep
        detour (angled but still broadly toward the goal) is preserved, only
        a target that would require net backward travel is excluded, letting
        the already-correctly-sorted fresh candidates get a real chance to
        win. Two regression tests added
        (`test_intermediate_targets_drops_a_stale_backward_pointing_last_target`,
        `test_intermediate_targets_keeps_a_last_target_that_is_still_a_reasonable_detour`).
        Verified live: the traced match now resolves with a goal scored, zero
        stall events (previously stalled for the rest of the 65s window).
      - **Full 231-match round-robin with the stale-intermediate-target fix
        (2026-09-04, `tournament_20260904_083049` vs. the brake-direction-fix
        baseline `tournament_20260904_075706`)**: stalled-match count dropped
        118 -> 87 (68 matches newly fixed, 37 newly stalled — net -31, the
        largest single-fix improvement this session). Spot-checked a newly-
        stalled match (`clear_danger_vs_tiki_taka`, 0-0 baseline with no
        stall, now `RESTART_STALL` at DIRECT_FREE_YELLOW) live: identical
        retreat/re-approach signature — closest friendly robot oscillates
        between ~2.3m and ~2.65m from the ball for 6+ seconds, never
        closing — so, per this session's established pattern with the brake
        fix, this reads as exposure to a DIFFERENT still-open instance of
        the same general congestion class (this fix only excludes a stale
        `last` more than 90 degrees off-axis; a fresh candidate itself
        oscillating, or a `last` that's stale but within 90 degrees, is not
        covered) rather than a new bug from this change. **Quality-signal
        picture is mixed here, unlike the brake fix's clean net-positive
        read**: shots 61 -> 69, turnovers 295 -> 329, completed passes
        560 -> 687, attacking-third entries 203 -> 237 (all up, consistent
        with robots making more real progress instead of idling in a
        planner-level local minimum) — but goals AND decisive matches both
        dropped, 21 -> 16. Not investigated further this session; worth
        checking whether this is restart-timeout variance (65s matches, a
        match that used to time out mid-approach might now complete the
        restart but not have enough remaining time to convert) before
        treating it as a real regression. Ship the fix regardless (it
        removes a genuine, confirmed planner defect with no legitimate case
        where retrying a >90-degree-stale cached target should ever win over
        a fresh, correctly-sorted candidate) but flag the goals/decisive dip
        for the next round-robin comparison rather than calling this
        unambiguously net-positive the way the brake fix was.
      - Remaining work: re-classify all 57 (now far fewer, exact count
        pending a full re-run against `tournament_20260904_083049` as the
        new baseline) DIRECT_FREE stalls against the FULL round-robin's
        actual before/after diff (not a margin-based static classification,
        which proved unreliable). The remaining 87 stalls are likely a mix
        of: still-open instances of the same general congestion class not
        covered by the 90-degree exclusion (see the goals/decisive dip
        above), the live-play `COMMITTED_FROZEN` ball-hold family (separate
        investigation, see the Handoff list), and BangBang1D's known defects
        (see below). Iteration is slow because the gate is the full
        231-match round-robin (~40 min on 15 workers); `--stop-at-first-stall`
        (done, see below) helps once a run is already known to contain a
        stall, but a fixed 30-40 match subset that reproduces each stall
        class (still open) is the real fix for iteration speed, precisely
        because a stall's reproduction depends on the full catalog/pairing
        context. `--fuzz-restarts SEED` (405693c) exercises restarts far
        more often than natural play and is the right way to bench a
        restart fix.
    - **Handoff, 2026-09-03 (open, in priority order):**
      1. Live-play ball holds above (three_slot / low_block). Being traced
         in a separate session, which suspects "converged-target churn":
         a marker whose `man_mark` target sits within cm of its position
         replans a sub-0.1 s trajectory every tick. Unverified; a robot at
         its target is *meant* to rest, so check the issued targets in
         `<run>/<match>.intentions.jsonl` for the carrier and marker before
         changing the planner.
      2. Trace the remaining 57 DIRECT_FREE stalls (single-match repro from
         the STALLS section; rsim is deterministic). Partial progress
         2026-09-04: one mechanism found and fixed (opponent-defense-area
         exemption never plumbed into `trajsample`, same class as the
         already-fixed `fpp` bug, `7a8e717`) but it only resolved 1/57 in
         the full round-robin — see the detailed writeup and the "confirm
         against the full round-robin, not a small subset" caveat above.
         A second harder mechanism (multi-robot congestion near a crowded
         defended corner) is identified but not yet traced to a fix.
         **Further progress, same day, second session**: live-traced (not
         replay-inferred — this is what let it be found at all, the earlier
         static/replay classification couldn't see it) a third, independent
         mechanism on the same match: `DirectFreeOursStep`'s kicker pick had
         no hysteresis and thrashed identity ~9x/second on ordinary rsim
         noise whenever candidates were near-tied, so no robot ever held the
         role long enough to progress. Fixed via `shared.tolerance.Sticky`
         (same primitive/margin `pass_and_shoot.py` already uses for the
         identical bug shape) — confirmed 243 -> 4 kicker switches on the
         same restart, two regression tests added. Does **not** fully
         resolve `clear_danger_vs_clear_press_plus` or a second spot-checked
         match (`three_slot_vs_tiki_taka_plus`) — both still `RESTART_STALL`
         at the same onset with this fix applied, because the congestion/
         local-minimum mechanism above is a separate, still-open problem
         that only becomes visible once thrashing noise is removed. Full
         writeup with live-trace numbers under "57 DIRECT_FREE stalls,
         first trace" above. Also flagged, unconfirmed: `BallPlacementOursStep`
         has the same unguarded `min(..., key=distance)` shape before ball
         arrival — worth checking against the 4 BALL_PLACEMENT stalls.
         **Third session, same day**: `BallPlacementOursStep` hardened with
         the same `Sticky` fix, unconfirmed whether it moves any stall count
         (see the dedicated writeup above). Separately, root-caused and fixed
         the emergency-brake direction bug in `TrajectorySamplingController`
         (scaling raw `robot.v` instead of the planned trajectory's
         velocity) — net-positive on every item-14 quality signal but raised
         the raw stall count (106 -> 118) by exposing more matches to the
         still-open congestion mechanism, confirmed via full round-robin
         diff. Then root-caused THAT congestion mechanism itself: a stale
         cached two-segment detour target in
         `TrajectorySamplingPlanner._intermediate_targets` was never
         re-validated against direction, only against staying collision-free,
         so a target picked once for a now-irrelevant situation (traced: 166
         degrees off the current goal direction) kept winning forever once
         the direct path became genuinely blocked — fixed by excluding any
         cached target more than 90 degrees off-axis. Full round-robin:
         stalled-match count 118 -> 87 (68 fixed, 37 newly stalled, net -31,
         the largest single-fix win this session); shots/turnovers/passes/
         entries all up, but goals and decisive matches both dropped
         (21 -> 16) — flagged as unresolved, possibly restart-timeout
         variance in the fixed 65s match window rather than a real
         regression, not investigated further this session. A newly-stalled
         match spot-check (`clear_danger_vs_tiki_taka`) shows the identical
         retreat/re-approach oscillation signature, consistent with exposure
         to a still-different instance of the same general congestion class
         (this fix's 90-degree exclusion doesn't cover every case) rather
         than a new bug. **Next step**: re-run the DIRECT_FREE stall count
         against `tournament_20260904_083049` as the new baseline, and
         investigate the goals/decisive dip before claiming this fix is
         unambiguously net-positive the way the brake-direction fix was.
      3. ~~Gate speed: stop-at-first-stall mode in `tournament.py`~~ Done
         2026-09-04: `--stop-at-first-stall` exits as soon as any match
         records a `StallEvent` (implies `--strict`; errors loudly if
         combined with `--no-save`, since stall detection needs
         `MatchResult.stats`, which `--no-save` never populates — a silent
         no-op would be worse than nothing there). Verified end-to-end
         (pre-fix `trajsampling/planner.py` via a temporary `git stash`,
         reverted immediately after — see the caveat above about not doing
         this concurrently with a background round-robin) that it correctly
         stops after the first stalling match and exits non-zero. The fixed
         30-40 match subset covering each stall class is still open.
      4. ~~`tools/metric_correlation.py` hardcodes `.pkl` replays (line ~429)
         and cannot read the current `.npz` runs.~~ Fixed 2026-09-04:
         `_iter_sampled_frames` now dispatches on extension like
         `replay_player.load_frames_in_range` already did, reading `.npz` via
         `ColumnarReplay.frame_at`/`n_ticks` and wrapping its
         `my_team_is_yellow` field in a `ReplayMetadata` so
         `compute_frame_metrics` needed no changes; `.pkl` still works
         unchanged for old run directories. `_worker` prefers `.npz`, falls
         back to `.pkl` when only that exists. Verified against both a
         current `.npz` run (`tournament_20260903_180026`, full 231-match
         pipeline including report generation) and an old `.pkl`-only run
         (`tournament_20260902_220936`).
      5. BangBang1D re-apply (`b26a550`) together with a blocked-start
         planner change - see the bullet below.
      6. Consolidate `tournament.py` / `full_match_tournament.py` /
         `arena_tournament.py` (item 14 prerequisite), then merge
         `tactic-engine` to `main`.
    - `BangBang1D.compute` has two real defects, pinned by the seeded sweeps
      in `tests/motion_planning/implementation/bang_bang_edge_cases_test.py`
      (marked xfail): a required-overshoot case (braking distance exceeds
      the gap, including p0 == p1 while moving) yields a negative phase time
      and a trajectory discontinuous in position and velocity, and a same-
      direction v0 > v_max case implies ~1e6 m/s² deceleration. A correct
      fix exists in `b26a550` and was backed out in `90d068c`: with
      physically correct trajectories, two robots parked ~0.5 m apart whose
      every candidate collides both stop and stay stopped, so a full round-
      robin stalled the opening kickoff in 231/231 matches (baseline 4).
      The discontinuous trajectories were letting mutually blocked robots
      creep through each other's paths. Re-apply `b26a550` only together
      with a planner change for the all-candidates-collide state (e.g. a
      short "yield" trajectory away from the nearest obstacle, or TIGERs'
      priority-ordered yielding applied to the fallback, not just to
      candidate rejection). Verify with `tournament.py --control-scheme
      trajsample --strict` and expect PREPARE_KICKOFF stalls ≤ 4.
    - **Sumatra-fidelity audit, 2026-09-04.** Requested comparison
      tournament (`trajsample` vs `fpp`, same-day/same-catalog) was blocked
      by repeated OOM kills on this machine (three attempts, 15/8/4 workers,
      all killed; `free -h` clean and no visible cgroup limit after each —
      most likely a `.wslconfig`-level VM memory cap, not a worker-count
      problem) and is parked, not abandoned — revisit once memory is free.
      Redirected instead to a subagent-driven line-by-line audit of
      `trajsampling/` against TIGERs' real Sumatra source
      (`github.com/TIGERs-Mannheim/Sumatra`, fetched via `gh api` rather
      than trusting a secondhand paraphrase), which surfaced 7 ranked
      findings; the top 3 were fixed and verified this session (each has a
      regression test confirmed to fail against pre-fix code via
      `git stash`):
      1. **Missing acceptor leniency** (`config.py`,
         `planner.py::_collision_leniency_accepts`). Sumatra's
         `MovingObstacleResultAcceptor.accept` doesn't reject every
         colliding candidate outright: a candidate already within
         `2*ROBOT_RADIUS` of its own final destination is accepted
         unconditionally (new early-exit in `plan()`), and — for a
         collision against a NON-priority obstacle only, a priority
         obstacle stays a strict unconditional reject — a collision within
         300mm of the destination is accepted if either the collision
         speed is under 1.5 m/s or the collision is further out than the
         robot's own current-speed braking time. Without this, a
         completely ordinary final approach next to a slow teammate/enemy
         was scored down by raw survival time exactly like a genuine
         head-on collision, pushing the planner toward `best_fallback`'s
         "whatever survives longest" pick even when a fine direct approach
         existed. Wired into both the direct-trajectory path and the
         two-segment fallback loop in `plan()`.
      2. **Escaping-grace mismatch between the collision scan and the
         emergency-brake clearance calc** (`planner.py::_with_current_clearance`).
         `collision_numba.py`'s per-obstacle escaping-grace state (an
         obstacle already penetrated at `t==start_t` gets one-time grace,
         revoked once cleared) was only honoured by the numba collision
         scan itself; `_with_current_clearance`'s `nearest_obstacle_distance`
         (used for the emergency-brake layer) re-minned across ALL
         obstacles including ones still being escaped, so a robot legally
         easing out of a stale penetration reported a large negative
         clearance and could trip emergency braking against an obstacle
         the collision-checker had already agreed to ignore. Fixed by
         excluding any obstacle the robot hasn't yet cleared (gap below
         the same dynamic margin formula used everywhere else) from the
         nearest-obstacle candidates, falling back to the unfiltered set
         only if every obstacle is still being escaped.
      3. **`_try_reuse`'s priority re-check only sampled 2 instants**
         (`planner.py::_try_reuse`). The re-check that lets a lower-priority
         robot notice a higher-priority teammate's freshly-replanned path
         crossing its own committed trajectory's margin only checked
         `elapsed` and `elapsed + MIN_TIME_STEP` (next ~20ms), unlike the
         general collision re-check just above it (`_first_collision`),
         which already scans the whole remaining trajectory. A robot could
         keep reusing a plan already known to walk into a priority
         obstacle seconds later, only noticing once `elapsed` itself
         finally caught up to that point. Widened to scan the full
         remaining duration (capped by `MAX_LOOKAHEAD_TIME`, same cap
         `_first_collision` uses) in `MIN_TIME_STEP` strides — a plain
         Python loop, not numba, since `_blocked_by_priority_obstacle` is a
         pure-Python per-instant check and this path prioritizes
         correctness over the hot-path speed numba buys `_first_collision`.
      - Full `motion_planning` suite green after all three
        (1198 passed, 84 xfailed, 273 xpassed, 0 failed).
      - Findings #4-7 from the same audit (lower-priority, not yet acted
        on): remaining minor Sumatra-fidelity gaps not yet triaged in
        detail — revisit if further planner hardening is warranted.
    - **PREPARE_KICKOFF_YELLOW regression from finding #3 above, found and
      fixed same session (2026-09-04), traced during a 15-worker OOM
      feasibility check** (`--max-workers 15` runs fine, memory flat at
      3.1-3.3 GiB/7.4 GiB throughout a full 231-match round-robin — that
      part of the check passed cleanly). The full-catalog run itself came
      back 231/231 stalled, 100% `PREPARE_KICKOFF_YELLOW`, vs. zero
      `PREPARE_KICKOFF_YELLOW` stalls in the prior 106/231 baseline —
      finding #3's widened re-check (scanning the WHOLE remaining
      trajectory up to `MAX_LOOKAHEAD_TIME`, 1.5s) was itself the
      regression: at a kickoff every non-keeper teammate simultaneously
      replans a multi-second approach every tick to converge on formation,
      none settling for more than a fraction of a second, so within a 1.5s
      window there is almost always SOME higher-priority teammate's
      (itself about to be replaced) trajectory crossing somewhere,
      permanently invalidating an otherwise perfectly good plan.
      - **Fix #1**: shrunk the re-check's own window to a new, dedicated
        `_PRIORITY_RECHECK_LOOKAHEAD_TIME = 0.5s` constant (was sharing
        `MAX_LOOKAHEAD_TIME` with `_first_collision`'s fresh-replan scan).
        Verified genuine improvement (BLOCKED-triggered invalidations went
        from nearly every tick to occasional bursts) but did NOT fully
        resolve the live stall on its own — the kicker still never
        converged to its kickoff spot in a live `debug_match.py` run.
      - **Fix #2, the actual root cause**: live-traced (with careful
        per-team obstacle/target filtering — a monkeypatched-dict-keyed-
        only-by-`robot_id` trace artifact briefly produced a misleading
        read by conflating yellow's and blue's same-numbered robots)
        that even with fix #1 applied, the kicker's direct line to the
        ball was priority-blocked on nearly every tick by some teammate's
        transient, about-to-be-replaced formation-approach path — the
        kicker is always the lowest non-keeper robot ID, so under
        `_has_priority`'s fixed "higher ID wins" ordering it is also the
        LOWEST-priority outfield robot, meaning every other teammate
        outranks it. Each block forced a `_two_segment_candidates`
        fallback through a freshly-random intermediate waypoint (traced:
        the switch-point location changed almost every replan, ~every
        0.2-0.3s), so the kicker never lived long enough on one two-segment
        plan to reach its own switch point and turn toward the real
        target — it just executed short first-leg bursts in place forever
        (confirmed: `has_collision=False` and correct target on every
        single `plan()` call throughout, yet position never converged).
        User-suggested and verified before fixing broadly: disabling
        `_blocked_by_priority_obstacle` entirely made the kickoff resolve
        cleanly and reproducibly — but priority-blocking is real,
        load-bearing protection during ordinary live play (module
        docstring: ported specifically to stop two teammates from grazing
        each other in the mirror_swap 6v6 test after four braking-only
        patches failed), so a global removal was rejected as too broad.
        Scoped instead to a new `_priority_blocking_enabled(game)` gate,
        threaded as a `priority_enabled` parameter through `plan()` ->
        `_try_reuse`/`_blocked_by_priority_obstacle` (both call sites):
        priority-blocking stays active during live play
        (`NORMAL_START`/`FORCE_START`, mirroring
        `utama_core.engine.match_stats`'s own `_LIVE_PLAY_COMMANDS`) and is
        disabled for every other referee command (every restart/formation
        phase, where mass simultaneous replanning makes the strictness
        counterproductive rather than protective). Ordinary (non-priority)
        collision avoidance — `_first_collision`,
        `_collision_leniency_accepts`, the emergency-brake layer — is
        untouched and still applies during restarts.
      - **Verified**: full `motion_planning` suite green (1209 passed, +10
        from 4 new regression tests, 84 xfailed, 273 xpassed, 0 failed).
        `debug_match.py` on the exact match that previously stalled
        100% of the time (`clear_danger_vs_high_press`, trajsample) now
        shows `stall_events: []` at both 20s and the full 65s tournament
        duration, with real gameplay (ball_travel 8.6m, 27 turnovers, 1
        completed pass over 65s) — not just the kickoff resolving in
        isolation. A same-day partial round-robin sample (35/231 matches
        so far) shows zero `PREPARE_KICKOFF_*` stalls; the remaining 8
        stalls are pre-existing `DIRECT_FREE_*`/`BALL_PLACEMENT_BLUE`/
        `COMMITTED_FROZEN` cases matching the categories already tracked
        above, not new regressions from this fix.
      - **Not yet done**: a full clean 231-match round-robin re-run to get
        a final, confirmed stall count comparable to the 106/231 baseline.
      - **Shipped** as `9757814`. Spot-checked afterward (2026-09-04,
        single-match `debug_match.py` reruns, not a full round-robin) against
        two of the three DIRECT_FREE_* stalls seen in the 15-match verification
        subset above: `clear_danger_vs_tiki_taka` and
        `clear_danger_vs_split_shape` both now finish their full 65s window
        with `stall_events: []`. Both are DIRECT_FREE restarts, which this fix
        also disables teammate priority-blocking for (the gate is keyed on
        referee command being `NORMAL_START`/`FORCE_START`, not specifically
        `PREPARE_KICKOFF_*`) — so at least these two instances of the
        DIRECT_FREE "congestion/local-minimum" family tracked earlier in this
        item were actually the same restart-mass-replan priority-blocking
        mechanism as the kickoff regression, not a separate still-open bug.
        Not yet a full-round-robin-confirmed count of how much of the
        remaining DIRECT_FREE backlog this resolves — the next full run
        should re-classify against this as the new baseline before assuming
        the congestion mechanism is fully closed.
      - **Wider spot-check (2026-09-04, same session, 3 more single-match
        `debug_match.py` reruns)**: extended the check to the most stubborn
        cases from the earlier congestion investigation above —
        `clear_danger_vs_clear_press_plus` (the one that survived THREE
        separate prior fixes: kicker-sticky, brake-direction, stale-
        intermediate-target, still stalling at the identical t=43.8s onset
        every time), `three_slot_vs_tiki_taka_plus` (the second match that
        also survived the kicker-sticky fix), and `clear_danger_vs_shadow_switch`
        (the match the stale-intermediate-target fix was originally traced
        and fixed on). All three now finish their full 65s window with
        `stall_events: []` — `clear_danger_vs_shadow_switch` even converts to
        a 1-0 finish with a real shot. 5/5 spot-checked DIRECT_FREE-family
        matches are now clean, including the single most-resistant repro
        case in this whole investigation. Reads as strong (not yet
        exhaustive) evidence that restart-scoped priority-blocking was the
        actual root cause underlying the whole congestion/local-minimum
        family, not a coincidental fix for 2 cases — the earlier fixes
        (kicker-sticky, brake-direction, stale-intermediate-target) were all
        real, necessary bugs, but priority-blocking during the mass replan
        was the mechanism that kept re-creating the same retreat/re-approach
        deadlock underneath them. Still only single-match spot-checks (5 of
        an original ~56-match backlog, chosen because they were the most
        heavily-documented, hardest-to-fix cases, not a random sample) — a
        full round-robin re-run remains the honest way to get a final count
        and rule out cherry-picking.
    - **`COMMITTED_FROZEN` live-play ball-hold, root-caused and fixed
      (2026-09-04, same session)**: the other liveness category from the
      handoff list above (`three_slot`/`low_block` defenders holding the
      ball). Root cause is in `DefenseTactic.tick()`
      (`utama_core/tactics/defense.py`), not the planner: its loose-ball
      retriever branch sends the nearest defender to `go_to_ball`, but the
      instant that robot's IR sensor (`has_ball`) actually goes True,
      `ball_is_loose(game)` flips False on the very same tick (any friendly
      `has_ball` makes it so) — so `retriever_id` resets to `None` next
      tick and the ball-carrying robot falls straight back into
      `defend_parameter`'s pure shot-shadow positioning, which has zero
      ball awareness. Nothing ever told it to release or clear the ball, so
      it just dragged the ball along its shadow path indefinitely —
      matches the roadmap's original live description exactly ("a
      `low_block` defender sliding along x = -3 with the ball glued to its
      dribbler"). `ClearBallTactic` already exists as the catalog's
      intended "kick the ball out of danger" relief valve, but `low_block`
      never wires it in (its own docstring frames it as the deliberately
      minimal two-Tactic baseline), and wiring it in wouldn't have helped
      anyway since the bug is inside `DefenseTactic` itself, not a missing
      Tactic slot. Fixed by giving `DefenseTactic` its own `carrier_id`
      cross-tick field (`DefenseMem`): checked directly via `has_ball`
      against every assigned robot every tick (not gated behind
      `ball_is_loose`/`retriever_id`, which can never observe the
      acquisition — by the time `has_ball` is True, `ball_is_loose` is
      already False on that same tick), it keeps a robot that already has
      the ball as the active (non-shadowing) one until it genuinely no
      longer has it, and routes it through a new `_clear()` method — the
      same chase-can't-happen/aim/kick sequence `ClearBallTactic` uses
      (reusing its shared `has_ball`/`oriented_towards` helpers), aiming at
      a single fixed upfield-and-away-from-goal target rather than
      `ClearBallTactic`'s multi-lane scoring, matching `DefenseTactic`'s
      existing "minimal baseline" role. Two new regression tests
      (`test_defense_carrier_stays_assigned_after_retriever_gets_the_ball`,
      `test_defense_carrier_releases_once_it_no_longer_has_the_ball`) pin
      both halves of the fix — the carrier must be picked up correctly AND
      released once the ball is actually gone, not traded for a different
      permanent-hold bug. Full suite green (4186 passed, 0 failed).
      Live-verified via `debug_match.py` (`high_press_vs_low_block`, one of
      the two matches this exact `COMMITTED_FROZEN` category was originally
      seen in): 65s window, `stall_events: []`; not yet confirmed the fix's
      carrier path was actually exercised in that specific run (no
      dedicated trace key was added, unlike `ClearBallTactic`'s
      `clear_ball[id]` trace) — the confidence here rests primarily on the
      unit tests and the direct code-level root-cause fix, not on a
      guaranteed live repro. A full round-robin re-run would give a
      before/after `COMMITTED_FROZEN` count, same caveat as the DIRECT_FREE
      spot-checks above.
    - **Two-segment-candidate instability, root-caused and fixed
      (2026-09-12)**: full round-robin re-run (231 matches, same seed/
      duration as every count above) confirmed 79 stalled (mostly
      RESTART_STALL) — the priority-blocking fix above closed most of the
      DIRECT_FREE backlog but left a distinct, still-open mechanism in
      `_intermediate_targets`'s "retry the previous winning detour" logic.
      Two separate bugs found and fixed in the same investigation:
      - **Bug 1 — stale absolute-point staleness check.** `last` was stored
        as an absolute point, re-checked each call by computing ITS bearing
        from the CURRENT `p0` against the current final-target direction.
        A robot near the 1m sampling ring's own centre can have that
        bearing swing 50-90+ degrees from a `p0` shift of only a few tens
        of centimetres, with no real change in the underlying situation.
        Live-traced on `clear_danger_vs_overload_flow`'s DIRECT_FREE_YELLOW
        stall: `last` measured 84.1 degrees off-axis (kept) one replan,
        then 92.1 degrees (dropped) the very next tick, purely from `p0`
        drift — every drop threw away a perfectly good detour and forced a
        fresh random draw, which itself went stale the same way a few
        replans later, so the robot never stayed on one detour long enough
        to clear the blocking obstacle (net displacement near zero for the
        rest of the match, same shape as the earlier stale-backward-target
        bug, just via direction churn instead of one frozen bad choice).
        Fixed by storing `last` as a unit DIRECTION from the robot instead
        of an absolute point, re-anchored onto a fresh
        `INTERMEDIATE_TARGET_RADIUS` ring around each call's CURRENT `p0`
        rather than reused as a stale coordinate — the staleness check is
        then invariant to `p0` drift alone; only a real change in which way
        progress lies moves it past the threshold.
      - **Bug 2 — side-flip oscillation once `last` itself fails.** Fixing
        bug 1 exposed a second, previously-masked instability: when a
        single near-stationary teammate sat almost exactly astride the
        direct path, `last`'s own switch-time search would occasionally
        fail (a few tens of millimetres of drift is enough to flip a
        borderline-clear detour to borderline-blocked), and the unbiased
        fresh-random fallback was exactly as likely to flip the detour to
        the OPPOSITE side of the same obstacle as to retry the same side —
        live-traced on a PREPARE_KICKOFF_YELLOW regression this fix itself
        introduced in `clear_danger_vs_overload_flow` (caught via a
        `git stash` before/after comparison, not by the round-robin): 18
        side-flips over 16s, net zero progress. Fixed by trying
        `_N_NEAR_LAST_JITTER` (3) same-side jittered variants of `last`'s
        own direction (±11°, ±22°, ±33°) before falling through to fully
        unbiased fresh draws — cut the same match's flip count to 2 over
        16s and resolved the regression.
      - **Bug 2b — moving-blocker variant jitter alone couldn't fix.**
        Spot-checking beyond the original repro found `counter_press_vs_
        {overload_flow,score_aware_zone_flow,zone_fluid}` all stalling at
        an identical PREPARE_KICKOFF_BLUE tick (confirmed via `git stash`:
        also a new regression from bug 2's fix, not pre-existing). Root
        cause: this case involves TWO other teammates simultaneously moving
        into their own nearby formation spots (not one near-stationary
        blocker) — the blocking geometry itself sweeps across the jitter's
        ±33° window faster than the search can track it, so the fallback
        still eventually hits the same unbiased-random side-flip as bug 2,
        just on a longer timescale. The actual fix was unrelated to
        widening the jitter search: `_collision_leniency_accepts`'s
        destination-proximity gate (leniency only applies within 300mm of
        the final target) had no restart-phase awareness, so a slow,
        transient graze against a still-moving teammate ~0.9m from the
        kicker's own destination was rejected outright regardless of how
        safe it actually was — inconsistent with priority-blocking already
        being unconditionally disabled for this whole referee-command
        family precisely because mass-simultaneous restart replans
        routinely cross paths transiently and harmlessly. Fixed by skipping
        the destination-proximity gate entirely when `priority_enabled` is
        already False (restart/formation phase) — the speed/braking-
        distance check (check 3) is unchanged, so a genuine fast head-on
        collision is never excused merely for happening during a restart.
      - **Verified**: full `motion_planning` suite green (1211 passed, +2
        new regression tests — same-side jitter candidate-list shape,
        leniency accepted/rejected under `priority_enabled=False` — 84
        xfailed, 273 xpassed, 0 failed). Two full 231-match round-robin
        re-runs: after bug 1+2 alone, 78/231 stalled (down from the 106/231
        original baseline, roughly flat vs. the mid-session 79/231 figure —
        net improvement, no material regression); after bug 2b's fix, a
        second full re-run came back at **27/231 stalled** — RESTART_STALL
        9 (down from 61), NO_PROGRESS_POSSESSION 2 (both `counter_press_vs_
        {press_trigger_flow,score_aware_counter_flow}`, confirmed pre-
        existing via `git stash`, not new), COMMITTED_FROZEN 16 (the
        pre-existing live-play tactic-freeze category tracked above,
        deliberately left untouched this session — different code path,
        `DefenseTactic`/tactic-decision logic, not the motion planner).
        Zero PREPARE_KICKOFF_* stalls of any kind in the final run. Shipped
        as `4b701ae`.

    - **`BALL_PLACEMENT_*` "double-axis corner overshoot" `RESTART_STALL`
      (20/231 at this point), root-caused and fixed 2026-09-12.** All 20
      remaining `RESTART_STALL` cases after the fix above shared one onset
      shape: a `BALL_PLACEMENT_*` restart that never advanced for the rest
      of the match, with the ball found resting exactly at
      `(±4.7785, ±3.2785)` — a field corner, past the boundary on both x
      and y simultaneously.

      Two false starts before the real cause, each round-robin-verified as
      net negative and reverted rather than shipped:
      - *Attempt 1* — `BallPlacementOursStep`'s out-of-bounds chase branch
        was changed to drive straight at an out-of-bounds ball's exact
        position (bypassing the normal behind-the-ball approach offset,
        which could point even farther out of bounds than the ball
        itself). Round-robin: 37 stalled (28 `RESTART_STALL`) vs. the 29
        (20 `RESTART_STALL`) baseline — 2 fixed, 8 new regressions, all one
        mechanism (see below).
      - *Attempt 2* — added `_field_boundary_exempt()` to the trajsample
        planner (`planner.py`), dropping the field-boundary
        `StaticSegmentObstacle` edge(s) from a `plan()` call's obstacle set
        when the target itself was out of bounds on that axis, mirroring
        the existing per-call ball exemption. Verified live to work exactly
        as designed at the obstacle-list level (for a corner target,
        correctly dropped 2 of 4 boundary segments, and the resulting
        `plan()` returned `has_collision=False` with a genuinely valid,
        non-degenerate trajectory) — but the robot's real position in
        `debug_match.py` traces still never changed tick-to-tick. Round-
        robin: 37 stalled (27 `RESTART_STALL`) again — 2 fixed, 9 new
        regressions, same signature. Both attempts reverted back to
        original/pre-session behaviour once the real cause (below) was
        found; neither was needed.

      **Root cause**, found by live-tracing ball position/velocity
      tick-by-tick across the whole pipeline rather than reasoning from the
      planner or referee-action layers: `strategy_runner.py`'s existing
      sim-only shortcut — teleporting the ball straight onto
      `designated_position` the instant `BALL_PLACEMENT_*` begins, since "a
      robot cannot physically retrieve an out-of-bounds ball in
      simulation" — was firing correctly, but the native rsim engine
      produces a large post-teleport velocity-spike artifact on the very
      next physics step (a known, already-documented issue;
      `_TELEPORT_SETTLE_TICKS`'s re-pin window exists specifically to
      absorb it). The window's release condition, however, only checked
      whether speed had decayed below `_TELEPORT_SPIKE_SPEED_MPS` (3.0
      m/s) — a threshold sized to classify "is this the reset artifact"
      (tens of m/s observed), not to mean "the ball is actually at rest".
      Traced live: one spike decayed from ~13 m/s to 0.83 m/s two ticks
      later, comfortably under 3.0, so the window released there — and
      since an SSL ball rolls with very little friction, that residual
      0.83 m/s carried it on a straight, slowly-decaying coast for ~14
      seconds and ~2.5m until it wedged into a field corner, never once
      reaching the placement target. `_ball_placement_done()` gates purely
      on ball-to-target distance, so this was an unbounded wait with no
      other way to become true — the actual mechanism behind every one of
      the 20 stalls, not a genuinely unreachable geometry (an independent
      Opus subagent investigation, run before this trace, had concluded
      the corner ball was outside the chassis-plus-contact-sensor
      reachable radius by ~0.016m and recommended clamping placement
      targets to a wall-reachable box — a real, defensible finding for
      *that* narrower geometric question, but not the actual cause of the
      stall, since the ball never should have been coasting into that
      corner in the first place).

      **Fix** (`utama_core/run/strategy_runner.py`): added a new, much
      stricter `_TELEPORT_SETTLE_SPEED_MPS = 0.05` that `_tick_teleport_
      settle`'s release condition now actually checks (`ball_speed <=
      0.05`), replacing the old check against the coarse spike-classifier
      threshold; the extension loop now simply extends for as long as the
      ball hasn't settled; `_TELEPORT_SPIKE_SPEED_MPS` removed as
      dead. Also added a robustness backstop in
      `utama_core/custom_referee/state_machine.py`:
      `_BALL_PLACEMENT_TIMEOUT_SECONDS = 10.0`, mirroring the existing
      `_STOP_CLEAR_TIMEOUT_SECONDS` pattern — Auto-advance 4 now auto-
      advances `BALL_PLACEMENT_* → next_command` anyway if the placement
      still hasn't succeeded after the timeout, recording
      `ball_placement_failures`/`can_place_ball=False` on the placing team
      (SSL rulebook §5.3.3 mirrors this: a GC operator eventually rules a
      stuck placement failed and hands the restart to the other team) —
      covers any other unreachable-target cause independent of the
      teleport-settle fix, including the narrower geometric case the Opus
      investigation surfaced. `BallPlacementOursStep`'s two genuinely-good
      fixes from earlier in the session (behind-the-ball `_APPROACH_OFFSET
      = 0.10` so the dribbler, not the chassis centre, reaches `has_ball`
      contact range; `_SETTLED_SPEED_MPS = 0.3` gating the release
      countdown on the ball actually having slowed down) were kept as-is.

      **Verified**: full referee/strategy_runner/motion_planning test
      suites green (9057 passed, 84 xfailed, 274 xpassed, 0 failed),
      including new regression tests for both fixes
      (`test_teleport_settle.py::test_settle_window_extends_through_
      residual_speed_below_old_spike_threshold`,
      `test_custom_referee.py::test_simulation_ball_placement_times_out_
      and_advances_anyway`). Full 231-match round-robin:
      **`RESTART_STALL` 20 → 1** (the one survivor,
      `give_and_go_solo_vs_switch_of_play`, stalls on `PREPARE_KICKOFF_
      BLUE` — an unrelated referee command/mechanism, not investigated
      this session). `COMMITTED_FROZEN` (deliberately out of scope, tracked
      separately above) 9 → 10, one new case
      (`decoy_and_overload_vs_high_press`) — noted, not investigated.
      Shipped as `43d41af`.

    - **2026-09-12 session: 4 more `COMMITTED_FROZEN` root causes**, picked
      up directly from the "noted, not investigated" case above. Each was
      confirmed by live-tracing the actual match (not reasoned about from
      the code alone), fixed, and individually re-verified stall-free before
      moving to the next:

      1. `turn_on_spot()` (`move_utils.py`): `_PIVOT_CLEARANCE_M` (exact
         body-touching distance, zero margin) still deadlocked live — an
         enemy parked 1.8mm outside the exact cutoff never tripped the
         guard, yet rsim's real contact resolution still resisted the
         commanded push at that range. Widened the clearance by 5cm (same
         style as `_pass_and_score.py`'s `_MIN_SETUP_CLEARANCE`) and, when
         blocked, steer `move()`'s own hold-position target away from the
         enemy instead of zeroing `local_left_vel` — the zeroed-push
         version measured live as literal zero net motion for the rest of
         the match (rotating around the ball inherently requires the
         chassis to sweep an arc; dropping only the model's own lateral
         estimate doesn't remove that physical requirement, so rsim's
         contact solver had to supply/oppose the sweep itself).
      2. `_pivot_target()` (`switch_of_play.py`): the x-clamp alone only
         guarantees the target's *endpoint* sits outside the own defense
         area, not the straight-line approach to it — when the target's x
         lands behind the box, a `back_y` inside the box's own y-span means
         the approach cuts through the box's near edge regardless of which
         side the approaching robot starts from. Push `back_y` outside the
         box's y-span too whenever this happens.
      3. `_pass_exec()` (`_pass_and_score.py`): `intercept_point()`
         deliberately projects the receiver's own *live* position onto the
         passer's aim line (so the receiver can walk into whatever line the
         passer is aiming down) — but feeding that continuously-recomputed
         point straight into `move()` every tick creates a feedback loop
         while the receiver is still approaching: its own motion shifts the
         target past `_TRAJECTORY_TARGET_TOLERANCE` (0.01m) on nearly every
         moving tick, forcing `TrajectorySamplingPlanner._try_reuse` to
         replan from scratch instead of continuing the committed
         trajectory. Traced live (`decoy_and_overload_vs_give_and_go_solo`):
         of 1182 `_try_reuse` calls, 514 returned "target changed",
         dwarfing the 58 genuine collisions and 118 priority-blocks
         combined. Fixed by snapping `intercept_pos` to a 5cm grid — well
         inside `at_target`'s own 0.08m arrival tolerance, so it never stops
         the receiver short of actually arriving.
      4. `SwitchOfPlayTactic`'s "assess" phase (`switch_of_play.py`): the
         carrier's `has_ball(visual=True)` check had no grace period, so a
         single-tick sensor flicker (the same rsim dribble-physics quirk
         `_BALL_RECOVERY_RADIUS`'s comment already documents for "relay")
         sent the carrier straight into `go_to_ball`, discarding its held
         position — and since `_pivot_target()` is a function of the
         carrier's own live position, every such flicker dragged the
         pivot's target along with the carrier's chase, a second instance
         of the same feedback-loop shape as (3). Traced live
         (`high_line_zone_vs_high_press`): the carrier visibly walked ~1m
         chasing the ball over the course of "assess", the pivot target
         sliding the same distance in lockstep, `_debounced_settled` never
         converging because the target itself never stopped moving. Fixed
         with a 10-tick grace period, same pattern as
         `_pass_and_score.py`'s `_SETUP_BALL_LOSS_GRACE_TICKS`.

      Also hardened `DecoyOverloadTactic`'s "lure" phase with a 6s timeout
      on waiting for a teammate's carrier to pass (releases the slot back
      to the picker instead of holding indefinitely) — a reasonable
      safety net in the same spirit as `_FINISH_TIMEOUT_TICKS`, though it
      turned out not to be the dominant mechanism in the case that
      surfaced it.

      **Verified**: each fix individually confirmed stall-free on its
      target match before moving on; the 13 stalls from the round-robin
      that followed fixes 1-2 were all individually confirmed resolved by
      fix 3 (10 batch-verified, 3 stragglers verified individually). Full
      test suite green (11791 passed, 0 failed) — one test
      (`test_move_utils.py::test_turn_on_spot_suppresses_pivot_push_into_a_
      wedged_enemy`) asserted the old zeroed-push behaviour and was updated
      to assert the new steer-away behaviour instead. Full 231-match
      round-robin `COMMITTED_FROZEN` trend across the session: 11
      (baseline) → 13 (after fixes 1-2, most newly-reachable since matches
      now run further before freezing) → 10 (after fix 3) → **8 matches, 9
      stall events** (after fix 4) — net progress, not yet zero.

      **Residual, not fixed this session**: the remaining stalls
      (`clear_press_plus_vs_overload_press`,
      `decoy_and_overload_vs_high_line_zone` ×2,
      `decoy_and_overload_vs_overload_flow`, `high_line_zone_vs_split_shape`,
      `overload_press_vs_three_slot`, `press_and_pass_vs_shadow_switch`,
      `three_slot_vs_tiki_taka_plus`) were traced one representative case
      in depth (`decoy_and_overload_vs_overload_flow`): the passer's
      orientation settles and `intercept_pos` stabilizes normally, but the
      receiver — boxed in by two closely-spaced enemies — never converges
      on it; its commanded velocity oscillates in sign rather than
      committing to one route, matching this codebase's own documented
      "two-segment candidate instability" bug class (`4b701ae`) rather than
      a tactic-logic bug. It is bounded: `DecoyOverloadTactic`'s existing
      12s `_FINISH_TIMEOUT_TICKS` self-resolves it (confirmed live —
      `committed_robot_ids` clears ~12s after "finish" phase starts), just
      slower than the stall watchdog's own threshold. A real fix belongs in
      `TrajectorySamplingPlanner`'s two-segment candidate selection, not
      tactic code — separate, larger work, not attempted this session. Also
      one new `RESTART_STALL` (`score_aware_zone_flow_vs_switch_of_play`,
      `DIRECT_FREE_YELLOW`) appeared, a different category untouched this
      session. Shipped as `70cb5c6`.
