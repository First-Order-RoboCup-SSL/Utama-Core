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

- **More tactics (first pass)** — `press_and_contain`, `give_and_go`, 5 example
  `Strategy` configs (`dd73bbc`).
- **BT/py_trees removal** — `AbstractStrategy` rewritten kernel-native, 13 BT
  example strategies + blackboard/tree scaffolding deleted, all riding tests
  ported to kernel-based strategies first (`087ee4b`, `960662c`). `strategy/
  referee/actions.py` + `abstract_behaviour.py` deliberately kept (load-bearing
  for `kernel.RefereeOverride`).
- **CI** — already existed (`.github/workflows/ci.yml`/`lint.yml`); found and
  fixed 2 pre-existing test bugs blocking a green run on `spike/tactic-kernel`
  once it was actually pushed.
- **Tournament scoreless-draw debugging** — 86% → 72% scoreless-draw rate via:
  rSim kick-direction physics fix (`docs/patches/rSim-kick-direction.diff`),
  two `FastPathPlanner` bugs (NaN divide-by-zero in `_find_subgoal`,
  ball-adjacent-obstacle target exemption), `SwitchOfPlayTactic` (new),
  `DefenseTactic`/`defend_parameter` foul-loop fix, `TwoDPID` braking-distance
  cap (`v <= sqrt(2*max_acceleration*error)`). `two_robot_attack` renamed to
  `pass_and_shoot`.
- **Motion-controller discontinuity handling** — `AbstractPID.calculate()` now
  auto-resets PID state on a target jump, orientation-only (`0.5 rad`
  threshold; translation deliberately left alone — see below). Removed 9
  manual `ctx.motion_controller.reset()` call sites this replaced (`91100ff`).
- **CustomReferee gaps** (double-touch rule, ball-speed rule, full-episode
  `reset()` for RL reuse, `set_bt_data` → `set_debug_status` rename) — all
  done; see `docs/custom_referee.md`'s "Known gaps" section.
- **Repo root cleanup** — stray committed session transcripts removed; 7 files
  importing deleted `strategy.examples` triaged file-by-file (2 ported:
  `demo_dribbler_test.py`, `main.py`; 5 deleted as having no kernel-tactic
  equivalent worth building).
- **Strategy-computation perf pass** — `FastPathPlanner._find_subgoal`
  bounding-box prune, `VelocityRefiner`/`KalmanFilter` scalar rewrites,
  `robosim` pipe I/O de-dup. ~3.0x end-to-end speedup measured (30s 6v6 match:
  ~47.5s → ~16.0s wall time), compounding across all perf commits this
  session (`3086337`, `0cf1e17`, `679e8cd`, `b79b863`, `b98cd29`, `4493002`).
- **`AGENTS.md`** — agent-agnostic contributor doc: kernel/`Tactic`/`Strategy`/
  `Partitioner` model, single-writer-partition invariant, minimalism
  discipline, headless/CI conventions, "Writing a Tactic" guide.
- **Goalkeeper overshoot (2026-08-24)** — two independent root causes, both
  fixed. (1) `predict_ball_pos_at_x` returning `None` right as a shot crosses
  the goal line snapped the keeper's target to the goal center for one tick;
  `goalkeep.py` now holds the ball's own position instead when within 0.5m of
  the line. (2) `FastPathPlanner`'s lookahead "carrot" (up to 1m ahead of the
  robot) was being used for `TwoDPID`'s braking-distance cap instead of the
  true target, so the cap never engaged until the last ~1m — confirmed
  against a real tournament replay (`counter_flow_vs_tiki_taka_plus_Lk.pkl`,
  t=13-17.5s: keeper velocity reversed sign 6+ times approaching a target
  that had already settled). Fixed via `TwoDPID.set_final_target()`; measured
  ~40% avg / ~60-65% worst-case overshoot reduction in the reproduction
  match. Residual overshoot (~0.17-0.26m) is now consistent with the robot's
  physical acceleration limit, not a logic bug.
  **Update 2026-08-26:** that residual turned out to still be a live logic
  bug, not just acceleration-limited overshoot — a sustained, undamped
  oscillation (~1.6s period, ~0.74m amplitude, never converging) traced live
  during the gap #6/#9 referee-rule tournament validation
  (`replays/gap6_validation_20260826_124653/counter_flow_vs_tiki_taka_RK.pkl`),
  with the ball completely at rest and the predicted goal-line intercept
  fixed to <1e-6 drift for 2+ seconds — i.e. not the shot-approach dynamics
  the 08-24 fix targeted, but a keeper that can't hold still at all against a
  static target. Isolated `TwoDPID` gains alone (same gains, same
  start/target, no rsim in the loop) converged cleanly, so the instability
  only reproduces through the full sim loop with `FastPathPlanner` (`"fpp"`,
  `StrategyRunner`'s default `control_scheme`) in the path — consistent with
  `TwoDPID.set_final_target()` only having patched the carrot/braking-cap
  interaction, not eliminated whatever in `FastPathPlanner`'s
  carrot/detour-side routing was driving the underlying oscillation.
  **Fixed** by sidestepping the planner for the keeper entirely rather than
  debugging its detour logic further:
  `GoalkeeperTactic` (`utama_core/tactics/goalkeeper.py`) now builds its own
  dedicated `PIDController` (cached in `GoalkeeperMem`) instead of using
  `ctx.motion_controller` — the keeper's task never needs obstacle-avoidance
  path planning (it holds a point on its own goal line, inside its own
  defense area, where no legal opponent/teammate should be routing through),
  so nothing is lost by skipping `FastPathPlanner` for this one tactic, and
  no other tactic's motion control is touched. Verified via a live
  before/after measurement in the exact traced conditions (ball parked
  during `PREPARE_KICKOFF`, `prepare_duration_seconds` set very high so the
  target is provably fixed): steady-state y-position range over the last 2s
  went from **0.385m (oscillating) to 0.000m (fully converged)** with the
  fix. New regression test:
  `tests/strategy_runner/test_goalkeeper_stability.py::test_goalkeeper_converges_on_static_target_without_oscillating`.
- **Stuck-match root causes (2026-08-26)** — found via a new offline
  detector (`utama_core/replay/stuck_detector.py`'s `find_stuck_windows`,
  see `docs/testing_gaps.md` gap #11) run over the gap #6/#9 validation
  corpus. Two genuine multi-hundred-second stuck states, two distinct root
  causes, both fixed:
  (1) `PressAndContainTactic`'s presser positioned relative to a tracked
  enemy's *own* position when that enemy didn't have the ball, rather than
  straight at the ball — when the tracked enemy was itself stationary (its
  own team locked in an unrelated all-defense posture), the presser's
  target never converged on a fully loose ball, and it sat parked beside it
  for 566s. Fixed: `PressAndContainTactic.tick()` now drives straight at
  the ball via `go_to_ball()` whenever `game.robot_with_ball is None`.
  (2) `GiveAndGoTactic`'s `_pass_exec` synchronized passer/receiver
  handshake had no timeout — if the receiver never became ready, the
  passer held the ball indefinitely (`is_committed()` stays `True` for as
  long as `receiver_id is not None`), observed holding for ~335s in one
  match. Fixed: a new `hop_ticks` counter on `GiveAndGoMem` abandons a hop
  past `_MAX_HOP_TICKS` (4s), falling through to the tactic's existing
  shoot-or-reposition fallback. New regression tests:
  `test_press_and_contain_goes_straight_for_a_fully_loose_ball` and
  `test_give_and_go_abandons_a_hop_that_never_completes`
  (`utama_core/tests/engine/test_all_tactics.py`).
- **Defense-area retrieval stall** — a robot legally retrieving a ball resting
  in the opponent's defense area during a stoppage (`DIRECT_FREE_*`,
  `BALL_PLACEMENT_*`) previously stalled ~0.25m short — `FastPathPlanner`
  treated the defense area as an unconditional obstacle with no exemption for
  a legal dead-ball retrieval. Fixed via
  `_enemy_defense_area_retrieval_exempt()`.
- **Dashboard rebuild** — `custom_referee/gui.py` replaced with a unified
  Live/Replay/Tournament dashboard (`utama_core/dashboard/`); sparse
  (change-only) tactic/referee event logging instead of dense per-tick
  duplication; ~97% redundant per-tick `trace()` calls deduped via
  `MatchLog.trace_if_changed()`. `dashboard_server.py` is the standing way to
  browse replays/tournaments without a live match.
- **Touchline avoidance + ball-placement-into-defense-area stall** — same
  routing gap as the ball-contest deadlock below, triggered by a static
  obstacle (field wall / enemy defense-area rect) instead of a robot:
  `ball_adjacent_obstacles` was only applied to target sanitization, never to
  `check_segment`/`smooth_path` routing, so a ball near a touchline or a
  `BALL_PLACEMENT_OURS` carry into the opponent's box both converged just
  short and stalled. Fixed via a routing exemption scoped to *static*
  obstacles only (never robots — that's the reverted case below) plus
  extending `_enemy_defense_area_retrieval_exempt` to cover carry-to-place,
  not just retrieval.
- **SSL rulebook §8.3/8.4 audit + 7 new referee rules** — Pushing, Crashing
  (both position/velocity-only, no `has_ball` — see `robot_contact.py`),
  Keeper Held Ball, Excessive Dribbling, Robot Stop Speed, Ball Placement
  Interference, stoppage-time Robot-Too-Close-To-Opponent-Defense-Area, plus
  a Multiple Defenders sanction fix (penalty kick + ball-touch gating, not
  occupancy + free kick). Foul-counter/yellow-card mechanism
  (`TeamInfo.increment_foul_counter()`) wired up for the first time — no
  prior rule incremented it. Built via 3 parallel agents; see
  `docs/testing_gaps.md` for what the merge process caught and what it
  didn't.
- **Referee restart-formation fixes + strategy override hook** (2026-08-26).
  `PrepareKickoffOursStep`/`PrepareKickoffTheirsStep` (`custom_referee/
  actions.py`) hardcoded "goalkeeper = robot id 0" and folded the keeper into
  the outfield kickoff formation — a real bug once `AbstractStrategy
  .goalkeeper_id` is ever non-zero (item 3 below), and wrong even at id 0
  (unlike `PreparePenalty{Ours,Theirs}Step`, which already read the real
  keeper ID off the referee packet). Both kickoff steps now do the same:
  read `ref.{yellow,blue}_team.goalkeeper`, exempt that robot from formation
  entirely (absent from `cmd_map`, not just excluded from kicker choice), and
  pass `clear_own_defense_area=True`/`clear_opp_defense_area=True` to
  `_clear_to_legal_positions` as a live legality net (previously only
  `StopStep` did this for kickoff-adjacent formations — no live check meant a
  bad ratio/non-standard field could silently place a robot in a defense area
  with nothing to catch it before `NORMAL_START` fires and immediately
  refouls). Companion fix in `AbstractStrategy.step()`: `GoalkeeperTactic` now
  ticks whenever the goalkeeper's robot ID is absent from `cmd_map`, not only
  when no override command is active — needed because the keeper is now
  legitimately absent from the kickoff steps' output and must fall through to
  real goalkeeper logic instead of freezing.
  Also added: strategy implementers can now override any restart formation
  (`RefereeOverride`/`Strategy`/`AbstractStrategy` all gained a
  `referee_overrides: dict[RefereeCommand, Callable[[Game, MotionController],
  dict[RobotId, RobotCommand]]]` — pass it to `AbstractStrategy(...)` and a
  registered command bypasses the built-in `*Step` entirely; unregistered
  commands are unaffected). Previously there was no extension point at all —
  the only documented customization path was editing `actions.py` directly.
  12 new tests (`test_kickoff_goalkeeper_exemption.py`,
  `test_referee_overrides_customization.py`) plus 4 pre-existing
  `test_referee_unit.py` kickoff tests fixed (they asserted the old buggy
  behavior). Full suite: 781 passed, 0 failed.
- **`robosim` native stdout polluting the JSON protocol pipe (2026-09-01,
  `f613411`)** — found while root-causing 5/40 "crashed" cells in a
  competitive-tier `full_match_tournament.py` run. `rc-robosim`'s native
  (C++) layer occasionally writes a plain-text diagnostic (e.g. `"turnover
  0.86 robot x: ... ball y: ..."`) straight to the process's real stdout fd
  via `printf`/`std::cout`, bypassing `sys.stdout` entirely — confirmed via
  `strings` on the installed `.so` and live process/pipe inspection
  (`/proc/<pid>/fdinfo`). That text shares the same pipe as
  `robosim_subprocess.py`'s JSON replies; one such line coming out *in
  place of* a tick's reply (not just interleaved before it) either raised a
  `JSONDecodeError` one process up, or — in a first, reverted fix attempt
  that looped skipping non-JSON lines unboundedly — silently deadlocked
  forever waiting for a reply that would never arrive (verified live via
  `ps`/`/proc` inspection: the subprocess was idle, zero bytes buffered on
  the pipe). Fixed at the source in `robosim_subprocess.py`: duplicate the
  original stdout fd before the native extension is even imported, repoint
  fd 1 at `/dev/null`, and route the protocol's own JSON writes through the
  untouched duplicate — verified directly with an isolated positive-control
  test (a raw `os.write(1, ...)`, mimicking exactly how the native layer
  writes, confirmed landing in the redirect target while the JSON pipe
  stayed clean). `robosim_wrapper.py`'s read side also now bounds its
  non-JSON-line skip loop (10 lines) instead of looping unboundedly, so any
  future instance of this bug class fails loudly rather than hanging.

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
   urgent:
   - (Resolved 2026-08-26: `KernelContext` renamed to `TickContext` —
     "kernel" wasn't disambiguating anything (most of `engine`/`strategy`
     already reads as kernel-something), while `TickContext` says exactly
     what it is and matches its pre-port name
     (`utama_strategy.functional.core.TickContext`). Global rename across
     all 29 referencing files (`engine/`, `tactics/`, `skills/`, `tests/`,
     `demo_dribbler_test.py`, `AGENTS.md`, `docs/STRATEGY_DEVELOPMENT.md`,
     `docs/tactic_model_design_decisions.md`); this file's own historical
     bullets below keep the old name since they describe past state.
     Reconsidering whether the class needs new responsibilities beyond
     `motion_controller`/`match_log` remains open, but the naming half of
     this item is done.)
   - (Resolved 2026-08-26: `goalkeeper_id` now has a real, load-bearing
     override path — the kickoff-formation goalkeeper-exemption fix reads
     the actual keeper ID off the referee packet rather than assuming 0, and
     the new `referee_overrides` hook gives strategy implementers a way to
     customize restart formations per-command. See "Done" above.)

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
   Three separable sub-problems:
   - **rsim ball-stickiness bug — fixed and shipped (resolved 2026-08-16,
     re-confirmed 2026-09-01).** Root cause found and fixed in
     `rc-robosim`/`vendor/rSim` (upstream C++, ODE physics):
     `SSLWorld::setActions()` never called `setDribbler(false)` (a one-way
     latch), and `unholdBall()` didn't actually move the ball clear of the
     kicker collision envelope. Both fixed in
     `docs/patches/rSim-dribbler-release.diff` (source: `vendor/rSim/`, see
     `FORK_NOTES.md`). The patched build has been built and installed over
     stock `rc-robosim` in `.pixi/envs/robosim` since 2026-08-16 — this
     section previously and incorrectly said it was reverted to stock and
     blocked; that was stale/wrong, corrected 2026-09-01 after independently
     verifying the installed artifact's provenance (`direct_url.json` points
     at a local skbuild wheel, not PyPI; the installed `.so`'s md5 is
     byte-identical to `vendor/rSim`'s own build-tree output; the source tree
     already contains all three patches).
     The previously-suspected blocker —
     `test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
     regressing under the patch (`dist_to_center≈0.24m` vs. the required
     0.75m) — was root-caused on 2026-08-16 (see `FORK_NOTES.md`'s "Known
     issue: test fragility" section) as a **test-assertion bug, not a
     physics or planner bug**: the test measured distance from a fixed
     field-center point instead of the live ball position, while
     `_clear_to_legal_positions` was already correctly using
     `game.ball.p`. The dribbler fix changes ODE contact-solver branching
     during an incidental early-tick ball touch, chaotically shifting the
     ball's resting position ~0.56m from center — the tested robot was the
     whole time correctly 0.797m from the *real* ball position, comfortably
     clear of the 0.75m keep-out radius. Fixed by asserting against
     `game.ball.p` instead of a fixed origin, matching the sibling
     ball-placement test's existing pattern. The `perp_dir /
     np.linalg.norm(perp_dir)` degenerate-vector theory floated at the time
     was independently re-investigated and refuted twice (2026-08-16 and
     2026-09-01): `rotate_vector()`
     (`utama_core/global_utils/math_utils.py`) is norm-preserving by
     construction, and `planner.py` already guards the true zero-norm case
     before reaching that line — the numpy warning seen in that run was a
     red herring, not this code path's real failure mode.
     Re-verified 2026-09-01: targeted test 11 passed; full suite (CI's exact
     invocation) 843 passed, 4 skipped, 2 xfailed, exit 0. Not confirmed
     whether this was ever the same issue as
     [[project_rsim_dribble_issues]], but at minimum consistent with it —
     worth checking whether that memory is now stale too.
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
    a one-line patch. Worth a dedicated investigation since it's a real
    behavior that could show up in an actual match with similar spacing.

    **2026-09-02: `DWAController` (the already-built, already-pluggable
    `"dwa"` control scheme, `utama_core/motion_planning/src/controllers/
    dwa_controller.py`) resolves this scenario outright.** Ran the exact
    `test_mirror_swap` geometry standalone via `StrategyRunner(control_scheme=
    ...)` with both `"fpp"` (the hardcoded default in every test and in
    `StrategyRunner.__init__`) and `"dwa"`: `fpp` reproduces the failure
    (0/12 robots reached, one pair collided at 0.180m — worse than the
    doc'd "10/12, no collision," possibly config drift since the 2026-08-15
    finding, but still a clear failure either way); `dwa` converges cleanly,
    12/12 robots reached, no collision (min separation 0.281m), reproduced
    byte-identical on a second run (sim is deterministic, no RNG). This is
    consistent with the *mechanism*: every documented `FastPathPlanner`
    failure in this doc (this stall, the wall dead-end fix, the ball-contest
    deadlock) is a subgoal/carrot artifact — the planner commits to a
    discrete waypoint that turns out to be a bad choice and has no way to
    reconsider mid-approach. DWA has no subgoal concept to get stuck behind:
    it re-samples the full feasible velocity space every tick against live
    obstacle state, so a bad choice self-corrects the very next tick instead
    of persisting as a fixed target. Not yet known: whether `dwa` handles
    every scenario `fpp` currently handles fine (repro tool:
    `mirror_swap_dwa_probe.py`, parametrized by `control_scheme`) — this is
    evidence DWA sidesteps one real, documented failure class, not a
    recommendation to swap the default. Switching `StrategyRunner`'s
    default control scheme, or running a comparative tournament pass with
    `dwa` across the existing test suite, is a bigger decision than a single
    bug fix and deliberately left open rather than done unilaterally.

11. **Gameplay bugs observed via dashboard Live view** (flagged 2026-08-24).
    User-observed watching a live `tiki_taka_plus` 6v6 match. Three of the
    original four are resolved (goalkeeper overshoot — see "Done" above;
    direct-free-kick retrieval — see "Defense-area retrieval stall" above,
    though worth re-verifying that fix fully covers the originally-reported
    symptom; touchline/defense-area placement stalls — see "Done" above).
    One remains open:
    - **Ball-contest deadlock** (traced 2026-08-25; believed resolved
      2026-08-26 via `PushingRule`, confirmed with a targeted regression
      test — not fixed at the `go_to_ball`/planner level, and deliberately
      so). Root-caused via a real replay
      (`counter_press_vs_tiki_taka_plus_Rk.pkl`, t=40.23-46.17s: a
      `GiveAndGoTactic` carrier held 0.11-0.34m from a stationary ball for
      5.9s of live play, orbiting rather than closing). Mechanism:
      `FastPathPlanner._path_to`'s `ball_adjacent_obstacles` exemption
      (added for an earlier, similar stall) only exempts a ball-adjacent
      opponent from *target* sanitization, not from `check_segment`'s path
      *routing* — so the last approach segment to a contested ball is never
      collision-free and the planner detours forever, sweeping the carrot
      around the opponent's `OBSTACLE_CLEARANCE` ring instead of closing the
      gap. Tried and reverted: exempting the same obstacle from routing too
      (mirroring the existing defense-area-retrieval exemption) does let the
      robot reach the ball, but then exposes a worse failure — both robots'
      dribblers register `has_ball` simultaneously and grind in place
      (bodies pinned at exactly `ROBOT_DIAMETER` apart, ball crawling
      ~0.3m/5s instead of frozen). That's a genuine 50/50-contest physics
      case with no possession-arbitration logic to resolve it at the planner
      level — reframed via SSL rulebook §8.4.1 ("Pushing": "if both robots
      are pushing each other with similar force, no team is at fault") as a
      referee-level no-fault state, not a `go_to_ball`/planner bug. (Note:
      the touchline/defense-area routing fix above deliberately does NOT
      apply here — it's scoped to static, non-robot obstacles only, for
      exactly this reason.)
      `PushingRule` (built as part of the §8.4 rules audit, see "Done"
      above) detects exactly this geometry — position/velocity-only via
      `robot_contact.py`, no `has_ball` — and its symmetric-force branch
      issues `STOP` (no-fault, `offending_teams=()`) followed by
      `FORCE_START` at the ball's position, matching the rulebook exactly.
      `STOP` is a real `RefereeOverride` command, so `StopStep` then
      actively drives the encroaching robot outside the 0.8m ball keep-out
      radius — physically separating the pinned pair, not just logging a
      foul. Verified end-to-end by
      `tests/custom_referee/test_ball_contest_deadlock.py` (new,
      2026-08-26): one test drives the exact pinned-pair geometry through
      the real `CustomReferee.step()` call path (not `PushingRule.check()`
      directly — closes `docs/testing_gaps.md` gap #1 for this scenario) and
      confirms `STOP` is issued; a second confirms `RefereeOverride`'s
      `StopStep` actually moves the encroaching robot's target outside the
      keep-out radius. This was previously unverified even after
      `PushingRule` existed — the one live tournament check
      (`docs/testing_gaps.md` gap #6) never happened to trigger Pushing at
      all, so nothing had confirmed the fix actually covers the originally
      traced scenario until now.

    All open items need an actual match trace (via the dashboard's Replay
    tab, or `debug_match.py` + a temporary `trace()`/print hook) before
    attempting a fix — root-cause from real per-tick state, not from the
    symptom description alone.

12. **Testing-gap follow-ups from the §8.4 referee-rules audit** — see
    `docs/testing_gaps.md` for full detail; (1), (2), (3), (5), and the
    Pushing part of (6) closed 2026-08-26. (1)-(3): 16 new tests
    (`test_ball_contest_deadlock.py`, `test_referee_rules_integration.py`,
    `test_foul_counter_end_to_end.py`, `test_referee_scan_order.py`). (5)
    was root-caused differently than first framed: not "7 rules missing a
    `game_frame is None` guard" but one test
    (`test_custom_referee_set_command_accepts_scripted_metadata`) calling
    `CustomReferee.step()` with `game_frame=None` when its own signature
    declares `game_frame: GameFrame` — no real caller ever does that. Fixed
    the test to pass a minimal real `GameFrame` instead, and removed
    `BallPlacementInterferenceRule`'s now-dead guard rather than propagating
    it to the other 3 rules. Full suite: 799 passed, 0 failed. **On hold,
    revisit later**: (4) mypy/pyright adoption — would have caught the
    original signature-drift bug for free, but needs its own investigation
    into how much of the existing codebase would fail a cold run. Still
    open: (6, partial) `keeper_held_ball`/`ball_placement_interference`
    still haven't fired in any live tournament run (integration-tested now,
    but not field-validated) — worth checking specifically next time a
    tournament produces a long defense-area ball hold or a ball-placement
    restart.

    **2026-08-26: pre-live-tournament restart-safety audit.** Before
    proceeding to that (6, partial) field validation, did a full pass over
    every referee override action (`engine/referee_override.py`,
    `custom_referee/actions.py`, `custom_referee.py`, `state_machine.py`, all
    13 rule files) looking for restart-sequencing bugs, command-transition
    edge cases, and rule-vs-override interactions. Found and fixed two —
    `docs/testing_gaps.md` gaps #7 and #8:
    - **Gap #7**: `RobotStopSpeedRule` could foul a robot for still being
      driven out of the ball keep-out zone by `StopStep`'s own clearing
      motion (2s grace clock could expire before the override finished
      physically moving a robot that started deep inside the zone). Fixed:
      the rule now exempts a robot from the speed check while it remains
      inside `BALL_KEEP_OUT_DISTANCE` of the ball, regardless of elapsed
      grace time.
    - **Gap #8**: HALT (e.g. `DefenseAreaStoppageRule`'s 2nd-foul
      escalation) has no auto-advance anywhere, by rulebook design — a real
      match needs a human referee/GC to resume it. But no automated
      tournament/sim harness (`StrategyRunner`, `*_tournament.py`) ever did
      that either, so a live sim run tripping a HALT-issuing rule would
      freeze forever with no test failure, just silent non-progress — worse
      than the "rule never fires" trap (6, partial) already worries about.
      Fixed: `StrategyRunner` now auto-resumes HALT to `NORMAL_START` after a
      5s grace period, but only when `sim_controller is not None` (sim-only —
      never short-circuits a real match with an actual human present).

    Both fixes are regression-tested
    (`tests/custom_referee/test_dribble_placement_stopspeed.py`,
    `tests/strategy_runner/test_referee_rsim.py::test_halt_auto_resumes_to_normal_start_in_sim`)
    and confirmed against the full suite. No other confirmed bugs found —
    the override dispatch table covers all 13 commands correctly and
    keep-out/defense-area geometry math (radius vs. diameter, sign,
    reference point) checked out. `KeeperHeldBallRule`'s dwell clock
    resetting on every stoppage was also noted as a plausible (non-bug,
    rulebook-matching) reason it's never fired in tournament play — relevant
    context for the (6, partial) field validation still to come.

13. **Trajectory-sampling planner (`trajsampling/`) — architecture-level
    follow-ups, 2026-09-02.** This session did a Numba speedup pass on
    `trajsampling` (whole-loop-batched `@njit` kernels for `_first_collision`
    and the obstacle-distance functions, `collision_numba.py`; a per-tick
    shared-obstacle cache in `planner.py` that preserves mid-tick
    trajectory-commit visibility between sequentially-planned teammates —
    see the session's own reasoning about why a naive tick-cache would have
    silently broken that), then fixed a real correctness bug in the
    underlying `BangBang1D` primitive (`d_kill` sign error causing an
    endpoint discontinuity for opposing initial velocity — see the "Done"
    pointer below). A second agent, working independently in the same
    session, built the first dedicated test coverage for this planner
    (`utama_core/tests/motion_planning/implementation/
    trajsampling_correctness_test.py` — bang-bang/`Trajectory2D` invariants,
    Numba-vs-Python kernel equivalence) plus a scheme-agnostic controller
    contract test (`utama_core/tests/motion_planning/contract/
    controller_contract_test.py`) and a black-box, planner-independent
    standardized-scenario benchmark (`utama_core/tests/motion_planning/
    standardized/` + `tools/motion_planning_benchmark.py` +
    `docs/motion_planning_comparison.md`), none of which existed before —
    `trajsampling`/`fastpathplanning`/`dwa` had no common comparison harness
    prior to this. All of the below assumes that harness as the way to
    validate any future change here; see `docs/motion_planning_comparison.md`
    for how to run it and what it deliberately does and doesn't measure
    (no single weighted "winner" score — pass/fail is a safety-and-completion
    gate, ranking is left to reading multiple metrics together).

    A parallel review (an external research-agent consultation, not this
    codebase's own analysis) compared this implementation against the TIGERs
    Mannheim 2024 champion paper's published trajectory-sampling design (the
    architecture this module is modeled on) and flagged several gaps, in
    roughly the order worth tackling them:

    - **Directional-tube enemy-obstacle model (attempted, reverted — start
      here).** `EnemyRobotObstacle` (`obstacles.py`) currently models a
      moving enemy's reachable region as an isotropic expanding circle
      grown from two `BangBang1D` profiles. TIGERs' paper uses a directional
      tube instead — reachable envelope grows mostly along the enemy's
      current heading, staying near robot-width laterally — specifically to
      reduce unnecessary detours around opponents the circle over-avoids.
      **Attempted 2026-09-02, reverted the same session.** A capsule-shaped
      implementation (with full Numba-kernel parity and 6 passing unit
      tests) benchmarked clean in isolation but surfaced two real problems
      head-to-head against the pre-change baseline: (1) rsim's sensor/filter
      jitter on a "stationary" enemy (~0.0006 m/s) was enough to pick an
      effectively random tube direction, doubling `static_slalom`'s
      completion time (5.3s → 11.4s) before a stationary-speed threshold
      fix; (2) even after that fix, the dense `mirror_swap` (6v6) scenario
      regressed from "0 collisions, doesn't fully converge" to a
      **reproducible 1.1mm collision at t=7.37s**, and `static_slalom`'s
      slowdown persisted via a second, unisolated mechanism. Reverted
      rather than ship a known regression. **Next attempt should**: size the
      lateral bound to stay strictly conservative relative to the old
      circle at small time horizons (e.g.
      `lateral = max(0.5*a_max*tc**2, some old-circle-derived floor at that
      tc)`), and debug `static_slalom`'s regression and `mirror_swap`'s
      collision as two separate problems rather than in the same pass — use
      the new benchmark suite's per-scenario cells to isolate each fix
      independently before combining them.
    - **`Trajectory2D` discards lateral (transverse) velocity.**
      `Trajectory2D` is a 2D-space wrapper around a single `BangBang1D` solve
      along the straight line from `p0` to `p1` — see `bang_bang.py`'s
      module structure and the correctness test's own
      `velocity_cross_products ≈ 0` invariant. Any velocity component
      transverse to that line is silently dropped at trajectory start,
      which can demand an instantaneous direction change — most risky right
      after avoiding another robot, switching intermediate targets, or
      leaving a curved/lateral maneuver. Frequent replanning masks this in
      practice but doesn't enforce acceleration limits at the discontinuity
      itself. This is an architecture-level change to the hot path this
      session just finished optimizing (both `BangBang1D`/`Trajectory2D` and
      the surrounding Numba kernels assume the current single-axis shape) —
      wants a concrete failure case (a replay showing a bad
      velocity-direction snap) before starting, not just the theoretical
      argument, and should be benchmarked against the full standardized
      suite before/after given the blast radius.
    - **Every trajectory ends at zero terminal velocity.** Simple, but
      forces accelerate-brake-to-zero-accelerate cycles for continuous
      motion (moving-ball interception, support-role repositioning,
      dribbling) — TIGERs identify this as a known limitation of their own
      earlier bang-bang system. Likely the single largest behavioral
      improvement available here, per the external review, but is coupled
      to the lateral-velocity item above (a nonzero terminal velocity with a
      transverse component needs the 2D state model fixed first to be
      meaningful) — do that item first.
    - **Fixed robot-ID priority ordering is a valid symmetry-breaker but
      strategically weak.** `robot_id > other_id` prevents two robots from
      both deciding the other yields, but can make the wrong robot yield
      (a ball interceptor yielding to a distant support robot, a
      ball-carrier yielding unnecessarily). A stable multi-key ordering
      (goalkeeper/restart safety > possession/interception urgency > tactic
      role > distance-to-target > robot ID as final tie-break) would need to
      live in the planner's priority policy, not a new scheduler
      abstraction — and is really a `[[project_tactic_model]]`-adjacent
      concern (role/urgency signals) more than a pure motion-planning patch.
    - **No explicit stop/yield planner result.** When every candidate
      collides, the planner currently returns the longest-surviving
      candidate and relies on the controller's residual emergency brake.
      Making "yield" a first-class planner return value (rather than an
      indirect side effect of velocity scaling) would apply when the first
      collision is imminent, every candidate is priority-blocked, or the
      robot has no valid route.
    - **Collision sampling is adaptive-timestep, not a formal swept/
      continuous check.** Distance-based adaptive stepping (matching the
      TIGERs paper) can still in principle miss a narrow collision event
      between samples for a fast-moving obstacle. Bounding timestep by
      relative speed in addition to distance, or an analytic
      time-of-impact calculation for the linear-relative-motion case, would
      close this. The external review rates this as higher-value than
      further raw speed optimization at this point — a correctness-margin
      question, not a performance one.
    - **Intermediate-target sampling is random, not geometry-guided.**
      Matches the TIGERs paper's own five-random-target approach (not
      wrong), but can produce poor samples in narrow corridors and makes
      rare failures hard to reproduce/diagnose. Adding a handful of
      deterministic candidates (tangents around the first blocking
      obstacle, obstacle-normal offsets, forward/lateral offsets, the
      previous winner) alongside the existing random samples, then ranking
      by progress/clearance/duration/continuity, would improve
      reproducibility without giving up the random exploration.
    - **Stale committed-trajectory reuse could validate more.** Current
      reuse check: same target, robot still near predicted position,
      trajectory still collision-free, priority checks still valid.
      Possible additions: compare actual vs. planned velocity, invalidate on
      a sharp obstacle-velocity change, use a target-distance tolerance
      instead of exact-tuple equality, invalidate when a new obstacle enters
      the swept corridor, cap maximum reuse age.
    - **Bigger-picture, not yet decided**: the standardized benchmark's own
      existing strict-`xfail`s already show DWA independently resolving at
      least one known `FastPathPlanner` local-minimum-class failure
      (`test_mirror_swap`, see item 10 above). The external review's
      suggestion of a DWA-as-short-horizon-recovery-mode hybrid is plausible
      but is a planner-selection architecture question, not a same-planner
      patch — don't start it speculatively unless the full-matrix findings
      below make a concrete case for it.
    - Only after the above: consider Ruckig or another jerk-limited
      multi-axis trajectory generator as a full primitive replacement — per
      the external review, the nearer-term weaknesses are in obstacle
      modeling and multi-robot coordination, not in the bang-bang algebra
      itself, so this is explicitly a last item, not a starting point.

    **Done this session**: `BangBang1D` endpoint-continuity fix for opposing
    initial velocity (`d_kill` sign error in both `compute()` and
    `state_at()`, `7370736`) — see the corresponding strict-`xfail` test that
    now passes unmarked in `trajsampling_correctness_test.py`.

    **Locked baseline, 2026-09-02 (git rev `7370736`).** Ran the full
    extended benchmark matrix (12 scenarios × 3 schemes,
    `tools/motion_planning_benchmark.py`) after the benchmark-suite extension
    above landed — first as a single-repeat sweep, then re-ran the 3
    failing/borderline scenarios (`mirror_swap`, `crossing`, `narrow_passage`)
    at `--repeats 5` to separate real behavior from single-run noise. Every
    repeated cell reproduced byte-identical sim time across all 5 repeats
    (rsim is deterministic here, no RNG in the loop) — every finding below is
    a confirmed, reproducible behavior, not flakiness. Full reports:
    `benchmark_results/motion_planning_20260902_215718.md` (full matrix, 1
    repeat) and `benchmark_results/motion_planning_20260902_220040.md`
    (3-scenario, 5-repeat confirmation).

    Pass rate: fpp 9/12, dwa 8/12, trajsample 10/12 — **no scheme sweeps the
    board**; each has distinct, real failure modes:
    - `mirror_swap` (dense 6v6): fpp collides (5/5), dwa passes cleanly
      (5/5, matching the long-documented `test_mirror_swap` resolution —
      item 10 above), **trajsample stalls at `sim_timeout` with 8/12 robots
      reached, 0 collisions (5/5)** — this is a new, previously-unconfirmed
      finding. It directly contradicts this session's earlier informal
      head-to-head kernel-strategy match (which found trajsample winning
      cleanly against fpp) and is consistent with the same
      `mirror_swap`-fragility the reverted directional-tube attempt above
      separately found. Command-jump proxy count is the smoking gun:
      13,471 jumps / 5,295 direction changes / 1,253 brake events in this
      one cell — roughly 5x the next-highest cell in the whole matrix — real
      thrashing, not just slow convergence. **New open item**: root-cause
      why trajsample stalls (not collides) in this dense geometry; the
      informal head-to-head match's different result likely comes from a
      differently-shaped scenario (real kernel-strategy play vs. this fixed
      6v6 mirrored-swap geometry) rather than either result being wrong —
      worth reconciling once someone picks this up.
    - `crossing` (2 robots, perpendicular): fpp and dwa both collide (5/5
      each); only trajsample passes (5/5) — the one scenario where
      trajsample is the unique safe choice among the three.
    - `narrow_passage` (0.24m gap): fpp and trajsample pass (5/5 each); dwa
      collides (5/5) — matches the newly-added strict `xfail` in
      `test_scenarios.py` exactly.
    - `static_slalom`/`grid_intersection`: dwa fails both (collision); fpp
      and trajsample pass both.

    **DWA controller cost is real and matches the user's own recollection of
    DWA getting laggy in live matches.** Across every scenario in the
    5-repeat confirmation, dwa's `MotionController.calculate()` mean/p95 is
    consistently the highest of the three schemes (e.g. `narrow_passage`:
    fpp 0.28/0.73ms, dwa 0.63/1.00ms, trajsample 0.25/0.41ms —
    `mirror_swap`/`crossing` show the same pattern, dwa 2-3x trajsample's
    cost). This is a 60Hz control loop computed per-robot; DWA's per-tick
    full-velocity-space resampling doesn't reuse anything across ticks or
    robots the way trajsample's committed-trajectory reuse does, and this
    cost is measured here on only 1-2 controlled robots per scenario — a
    full 6-robot team is a plausible multiplier this benchmark doesn't
    directly exercise yet (all current scenarios control at most 2 robots
    per side; `mirror_swap` controls 6, but the per-cell latency numbers
    above are already visible even there). This directly informs the
    "switch the default scheme" question in item 10 above: DWA is not a free
    upgrade over fpp even where it's safer — it trades collision-robustness
    in specific geometries for a real, consistent per-tick compute cost
    increase, which likely compounds at full 6v6 scale into exactly the
    real-match lagginess previously observed. Any future default-scheme
    decision should weigh this directly, not just pass/fail rate.

    **Net effect on priorities above**: the `mirror_swap`/trajsample stall
    is now a second concrete trajsample gap (alongside the reverted
    directional-tube attempt) worth investigating before assuming
    trajsample is a strict improvement over fpp — the lateral-velocity and
    terminal-velocity items further up may or may not be related (a
    stalled, thrashing dense scrum is exactly where discarding lateral
    velocity would bite hardest), so whoever roots-causes the `mirror_swap`
    stall should check whether it's the same underlying mechanism before
    treating them as two separate fixes.

    **`block_shape.py` `Vector2D`/tuple type-contract bug, fixed 2026-09-02.**
    Found while running the first real competitive-strategy tournament under
    `control_scheme="trajsample"` (`tournament.py`, which previously had no
    `--control-scheme` flag at all — added this session, defaults to `fpp`,
    unaffected). `BlockShapeTactic.tick()` (`utama_core/tactics/
    block_shape.py`, used by `counter_flow`'s and other strategies'
    defensive screen) called `go_to_point(..., (lead_x, lead_y))` and
    `go_to_point(..., (screen_x, target_y))` — a raw tuple where the
    signature declares `Vector2D` — at two call sites. FPP and DWA's
    controllers happened to tolerate this silently; `TrajectorySamplingController.
    calculate()`'s stricter `target_pos.x`/`.y` access crashed outright
    (`AttributeError: 'tuple' object has no attribute 'x'`), which is what
    surfaced it. Fixed by wrapping both call sites in `Vector2D(...)`.
    Verified: the block_shape/all_tactics test subset (44 passed) and the
    full suite (870 passed, 4 skipped, 1 xfailed) both clean before and
    after.

    **Ball treated as an unconditional collision obstacle for the fetching
    robot itself — found and fixed 2026-09-02, the actual reason trajsample
    scored zero goals in every match it ever played.** After the
    `block_shape.py` fix unblocked the tournament, all 36 matches across the
    full competitive-tier round-robin (`counter_flow`, `tiki_taka`,
    `zone_fluid`, `tiki_taka_plus`, `score_aware_zone_flow`,
    `score_aware_counter_flow`, `clear_press_plus`, `shadow_switch`,
    `overload_flow`) came back 0-0 — not merely goalless, but with
    `ball_travel_m: 0.02` in every single match (the ball moved ~2cm total
    across a full 65s match) and `has_ball` never `True` once, despite
    robots repeatedly closing to within ~0.13m of the ball before visibly
    backing away instead of completing contact. Traced via
    `render_window()` (per `docs/STRATEGY_DEVELOPMENT.md`'s Observability
    section) plus direct `has_ball`/distance checks on the replay frames
    (per the standing practice of never trusting a rendered image alone for
    root-causing a stuck/frozen state) — root cause: `_shared_obstacles_for_tick`
    (`utama_core/motion_planning/src/trajsampling/planner.py`) added the
    ball as an unconditional `ConstantVelocityObstacle` (radius 0.0215m) for
    every robot's obstacle set, with no exemption for the robot whose
    current target IS the ball. `go_to_ball` (`utama_core/skills/src/
    go_to_ball.py`) deliberately targets a point slightly PAST the ball's
    centre (`_DRIBBLE_OVERSHOOT_M`) so the robot's motion controller keeps
    driving until actual physical contact — its own module comment says
    this overshoot exists "so the DWA keeps driving until the robot makes
    contact," an assumption that had never been checked against trajsample
    before this session, since trajsample had never previously been run
    through a real ball-fetching strategy (only synthetic point-to-point
    scenarios: `mirror_swap`, the standardized benchmark suite, and this
    session's earlier informal head-to-head kernel-strategy match — which in
    hindsight almost certainly hit the exact same bug, unnoticed because
    that comparison never checked `ball_travel_m`/`has_ball` directly).
    Under trajsample, every candidate trajectory toward that overshoot point
    necessarily collides with the ball's own collision circle before
    reaching it, so `_first_collision` reports a collision on essentially
    every candidate and `plan()` falls back to "whichever candidate merely
    survives longest" instead of a clean approach — which in turn starves
    `TrajectorySamplingController`'s residual closing-speed emergency brake
    into treating a completely normal ball-approach as an imminent
    collision, producing the observed approach-then-retreat pattern.

    **Fixed** by excluding the ball from the always-included per-tick shared
    obstacle cache and re-adding it back in per-`plan()`-call, per-robot,
    only when that call's own `target_pos` is farther than
    `_BALL_TARGET_EXEMPTION_RADIUS` (0.3m, generous relative to
    `go_to_ball`'s own sub-robot-radius overshoot distances) from the ball's
    current position — so a robot fetching the ball plans straight through
    it as intended, while every other robot (including one routing near a
    ball an enemy is actively dribbling) still treats it as a real,
    priority-respecting obstacle. New fields `_shared_ball_row`/
    `_BALL_RADIUS`/`_BALL_TARGET_EXEMPTION_RADIUS`; `_shared_obstacles_for_tick`'s
    docstring and `plan()`'s own inline comment both explain the mechanism
    for a future reader. Verified: full `motion_planning` suite unchanged
    (73 passed, 4 xfailed, byte-identical to pre-fix) — this fix only
    changes behavior when a target is near the ball, never touched by any
    existing test; a smoke-tested single match went from
    `ball_travel_m=0.02` (pre-fix) to `ball_travel_m=1.9` with `has_ball`
    firing 3,549 times (post-fix); the full 36-match tournament re-run
    confirmed this generalizes across every strategy pairing —
    `ball_travel_m` ranged 1.07–15.63m (mean 5.48m) instead of a flat 0.02m,
    possession splits became matchup-dependent instead of a fixed ~97.6%/
    2.4% pattern, and 2 real shots were recorded (0 before).

    **Still open, found while verifying the fix above — a real, separate
    second bug, not yet root-caused.** The re-run tournament still scored
    0-0 in all 36 matches despite the ball now genuinely moving and being
    contested. Tracing one match directly (`counter_flow_vs_tiki_taka.pkl`,
    the same pairing smoke-tested above): one robot (`friendly` id 1)
    registers `has_ball=True` for 3,544 of 3,601 frames in the 60s of live
    play — essentially the ENTIRE match, continuously, with only 7 total
    possession transitions recorded across the whole game and 0 shots taken
    by either side. This is not healthy give-and-go possession (which
    should show many short holds/passes) — it looks like the ball carrier
    permanently locks onto the ball and never transitions to a
    shooting/passing phase at all under trajsample, which fully explains why
    every match is scoreless even now that the ball is genuinely in play.
    Not yet investigated: whether this is (a) a tactic-level phase-transition
    condition (e.g. a shot-readiness or pass-decision check) tuned against
    FPP/DWA's carrot-following motion profile that never triggers against
    trajsample's direct-velocity output, (b) the dribble-overshoot geometry
    itself now keeping the robot glued in permanent contact rather than ever
    clearing the dribble sensor's threshold, or (c) something else entirely
    — genuinely unknown, flagged rather than guessed at. Whoever picks this
    up should start from the same `counter_flow_vs_tiki_taka` replay
    (regenerate via `tournament.py counter_flow tiki_taka --control-scheme
    trajsample`) and trace the carrier robot's tactic/phase state
    (`.intentions.jsonl`) alongside its `has_ball` timeline, the same method
    that found this. Until this is fixed, trajsample should not be treated
    as competitively viable for real strategy play even though its
    point-to-point motion planning (per the benchmark suite above) is
    otherwise reasonable.

    **`Trajectory2D.compute`'s degenerate zero-distance fallback, fixed
    2026-09-03.** Found while investigating a live user report ("robot gets
    to the ball and waits 5-10s before doing anything") against a
    `tiki_taka` vs `tiki_taka_plus` trajsample match. `move()`/
    `turn_on_spot()` (`utama_core/skills/src/utils/move_utils.py`) call
    `motion_controller.calculate(target_pos=robot.p, ...)` — target equal to
    the robot's OWN current position — every tick while orienting-in-place,
    e.g. `GiveAndGoTactic`'s pre-kick aim step. `Trajectory2D.compute` had a
    `dist < 1e-9` branch for exactly this case that picked a fixed, arbitrary
    axis `(1.0, 0.0)` to project `v0` onto, rather than a real "come to rest
    from current velocity" plan. Any residual velocity perpendicular to that
    arbitrary axis (e.g. all of it, if the robot's actual motion was purely
    lateral — the normal case for a robot pivoting on the ball) was silently
    dropped: the commanded velocity for that whole trajectory came out as
    exactly zero regardless of how fast the robot was actually still moving,
    so the planner never actually commanded a stop, only appeared to.
    **Fixed** by using the direction of `v0` itself as the projection axis
    when `dist < 1e-9` (falling back to the old arbitrary `(1.0, 0.0)` only
    when `v0` is also ~zero, where direction is moot) — this makes the
    single-axis `BangBang1D` solve see the robot's FULL speed rather than an
    arbitrary component of it, producing a real deceleration-to-rest profile.
    Verified: `motion_planning` suite unchanged (73 passed, 4 xfailed,
    byte-identical), full repo suite clean (943 passed, 4 skipped, 5 xfailed
    — all pre-existing/documented). Direct repro check before/after:
    `Trajectory2D.compute(p0=(1,2), v0=(0,0.8), p1=p0, ...)` previously
    commanded `(0, 0)` velocity for the entire trajectory despite 0.8 m/s of
    real lateral motion; now correctly ramps that velocity down to zero over
    a real ~0.2s braking profile.

    **However — re-running the exact reported match afterward did NOT
    reproduce the "waits at the ball" symptom, and traced to something new.**
    rsim is fully deterministic given unchanged code (three fresh
    `tiki_taka` vs `tiki_taka_plus` trajsample re-runs were byte-identical
    to the frame), so this is a real, distinct finding, not noise: from
    ~21s to the 65s match end, TWO robots on the SAME team (this match:
    enemy/`tiki_taka_plus` robots 2 and 4) simultaneously register
    `has_ball=True`, while the ball itself barely moves (~0.07m of drift
    total, not real carrying/dribbling) — a same-team scrum/pileup on a
    loose ball where a second robot converges on and "claims" a ball a
    teammate is already holding, and neither yields or actually drives play
    forward. This looks like the real mechanism behind both the user-visible
    long stalls and last session's "permanent ball-lock" finding above
    (hypothesis (a)/(b)/(c) there was framed as single-robot; this suggests
    it's actually a multi-robot allocation/contact problem instead — e.g. a
    picker or `go_to_ball` fallback that lets a second robot target a ball
    already legally possessed by a teammate). Not yet root-caused. Repro:
    `pixi run python tournament.py tiki_taka tiki_taka_plus --control-scheme
    trajsample --sequential --verbose`, then inspect
    `replays/tournament_<ts>/tiki_taka_vs_tiki_taka_plus.pkl` frames around
    t=21s for `enemy_robots[2]`/`enemy_robots[4]` — both within IR contact
    range of the ball (~0.10-0.11m) at the same time, ball position nearly
    frozen. Whoever picks this up should check which tactic(s) control both
    robots at that point (`.intentions.jsonl`) and whether `go_to_ball`/the
    picker has any guard against sending two robots at an already-possessed
    ball.

    **Root-caused and fixed, 2026-09-03.** Tracing the swapped-side match
    directly (`debug_match.py --strategy build_tiki_taka_plus_kernel_strategy
    --opponent build_tiki_taka_kernel_strategy`, so `tiki_taka_plus`'s own
    tactic decisions are traced) found the mechanism precisely:
    `KernelSchedulerStrategy` legitimately runs `GiveAndGoTactic` ("attack")
    and `DecoyOverloadTactic` ("overload") concurrently on disjoint robot
    subsets whenever we have the ball, but neither tactic has any way to
    know the OTHER one's carrier already has it. `DecoyOverloadTactic` picks
    its own "decoy" as whichever of its own two assigned robots is nearest
    the ball, with zero awareness of the rest of the team, then sends it
    straight there via `go_to_ball` the moment it doesn't already have the
    ball itself (`has_ball(game, mem.decoy_id)`, necessarily scoped to that
    one robot). One tick after `GiveAndGoTactic`'s carrier legitimately
    fetched the ball, `DecoyOverloadTactic` independently initialized in the
    same tick and sent its own decoy at the same ball, driving straight
    into the carrier — the scrum.

    **Fixed** with a new `_teammate_already_has_ball(game, excluding_id)`
    helper in `decoy_and_overload.py`: true when `game.robot_with_ball`
    points at a friendly robot other than the one asking, OR the ball is
    moving faster than `_LOOSE_BALL_SPEED` (0.3 m/s, matching
    `ball_is_loose`'s own threshold) — the second clause exists because a
    ball mid-pass between two other teammates is legitimately held by
    nobody for the handful of ticks it's in flight, and the first, cheaper
    check alone let a second scrum through exactly there. Wired in at
    every point this tactic decides to fetch the ball on its own initiative
    (the `len(robot_ids) < 2` degenerate fallback, and the "lure" phase's
    own `go_to_ball` branch — holding at a support point instead when a
    teammate already has it) plus a deeper fix: the lure-to-finish phase
    transition (`_LURE_MAX_TICKS` timeout) previously fired regardless of
    whether the decoy had actually collected the ball, and the "finish"
    phase immediately treats `mem.decoy_id` as the PASSER in the shared
    `_pass_exec` helper — which has its own unconditional `go_to_ball` call
    the instant that passer doesn't have the ball, completely bypassing the
    guards just added. A lure that timed out without ever fetching the ball
    (because a teammate elsewhere already had it) now simply keeps holding
    instead of transitioning into a phase built on the assumption that it
    already has the ball.

    Also fixed in passing: `debug_match.py` had no `--control-scheme` flag
    (unlike `tournament.py`), which blocked tracing one side's tactics
    under trajsample directly — added, matching `tournament.py`'s own flag.

    Verified: `decoy`/`overload`-scoped tests unaffected (13 passed), full
    repo suite unchanged (943 passed, 4 skipped, 5 xfailed, all
    pre-existing). Direct before/after on the reported match
    (`build_tiki_taka_plus_kernel_strategy` vs `build_tiki_taka_kernel_strategy`,
    traced via `debug_match.py`): the `go_to_ball[5]` call that used to fire
    the instant a teammate collected the ball now only fires once that
    teammate's pass has genuinely gone loose (ball speed measured dropping
    from 4.8 m/s to below the 0.3 m/s threshold with no receiver catch) —
    confirmed by direct ball-speed instrumentation, not just the absence of
    the old symptom. Re-running the original `tiki_taka` vs `tiki_taka_plus`
    tournament match: possession went from a stuck 15%/85% split to 70%/30%,
    and `ball_travel_m` roughly tripled (3.5m -> 9.7m) with possession now
    changing hands multiple times at realistic (single-digit-second) hold
    durations instead of one 40+ second frozen scrum. Matches are still
    scoreless — the separate "permanent ball-lock"/no-shot-transition bug
    documented above is a distinct, still-open issue — but the specific
    same-team-collision mechanism reported live ("gets to the ball and
    waits 5-10s") is fixed.
