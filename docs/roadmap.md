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

15. **`trajsample` liveness floor: 106/231 matches still stall (57 in a
    DIRECT_FREE restart, 42 live-play ball holds), and the BangBang1D fix
    cannot land until the planner handles blocked starts.** Findings from
    2026-09-03, all measured with the stall watchdog (`tournament.py` STALLS
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
      - Remaining work: re-classify all 57 (now 56) DIRECT_FREE stalls
        against the FULL round-robin's actual before/after diff (not a
        margin-based static classification, which proved unreliable), then
        trace the congestion mechanism above as its own investigation.
        Iteration is slow because the gate is the full 231-match round-robin
        (~40 min on 15 workers); `--stop-at-first-stall` (done, see below)
        helps once a run is already known to contain a stall, but a fixed
        30-40 match subset that reproduces each stall class (still open)
        is the real fix for iteration speed, precisely because a stall's
        reproduction depends on the full catalog/pairing context.
        `--fuzz-restarts SEED` (405693c) exercises restarts far more often
        than natural play and is the right way to bench a restart fix.
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
