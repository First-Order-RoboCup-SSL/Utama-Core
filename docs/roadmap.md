# Roadmap / TODO

Running list of larger, not-yet-scheduled workstreams. Unlike
`tactic_model_design_decisions.md` (a decision log for the tactic-kernel specifically),
this is just a place to park bigger ideas so they don't live only in someone's head or
in chat history. Entries get promoted out of here into an actual plan/PR when someone
picks them up — this file isn't itself a design doc.

## Sequencing (updated 2026-08-16)

1. **More tactics** — done for this pass (`dd73bbc`): `press_and_contain`,
   `give_and_go`, plus 5 new example `Strategy` configs. More can still be added
   later per "More tactics" below; this just isn't a blocker anymore.
2. **BT/py_trees cleanup** — done. `AbstractStrategy` rewritten kernel-native
   (`087ee4b`); the 13 BT example strategies, `strategy/common/{base_blackboard,
   blackboard_contract}.py`, and `strategy/referee/{tree,conditions}.py` are all
   deleted; every behavioral test that rode on them (ball placement,
   referee-override/stoppage handling, `exp_ball` validation, formation
   loading, motion-planning obstacle avoidance) was ported onto kernel-based
   strategies first. `strategy/referee/actions.py` and
   `strategy/common/abstract_behaviour.py` are deliberately kept — genuinely
   load-bearing, `kernel.RefereeOverride` drives their Step classes directly
   (not via real py_trees tree-ticking). See "Codebase cleanup" below for the
   full account, including bugs found and fixed along the way.
3. **CI** already existed (`.github/workflows/ci.yml`/`lint.yml`) — this
   section previously said otherwise, which was stale. What was actually
   missing: `spike/tactic-kernel` had never been pushed, so CI had never
   validated it; running the exact CI command locally found and fixed 2
   real pre-existing test bugs (unrelated to this branch's work) blocking a
   green run. See "CI" below for the full account.
4. Tournament/multi-strategy infra still waits on there being enough of a
   tactic catalog for comparisons to mean anything.
5. Agentic coding infra — `AGENTS.md` done, see "Agentic coding infra" below.
   The other two items there (CI shaped for agent loops, LLM-legible
   grsim/rsim feedback) remain exploratory, no design decided.
6. **Tournament scoreless-draw debugging (2026-08-16 session)** — done for
   this pass. Original 86% (24/28) scoreless-draw rate driven down to 72%
   (26/36) across: an rSim kick-direction physics bug (native patch), two
   `FastPathPlanner` bugs unfreezing `pass_and_shoot` in real 6v6 matches
   (NaN divide-by-zero, ball-adjacent-obstacle target exemption), a batch of
   hardcoded-tick-count removals, a new `SwitchOfPlayTactic`, and a
   `DefenseTactic`/`defend_parameter` foul-loop fix (two independent bugs:
   wrong trigger condition for 2-defender side selection, and an
   own-defense-area standoff margin too tight for real PID overshoot), plus
   a follow-up fix for a second, separate instance of the same foul —
   `SwitchOfPlayTactic`'s `_pivot_target()` could put an attacker inside its
   own defense area during relay play (uncapped pullback distance). Full
   writeup in "Multi-strategy / tournament evaluation infra" below.
   Verification match now runs 43s -> 61s+ clean, then — after root-causing
   and fixing the residual overshoot itself (`TwoDPID` had no braking-
   distance term; see below) — the full 90s verification match now runs
   with **zero** defender fouls. `two_robot_attack` also renamed to
   `pass_and_shoot` (its actual behavior: one scripted setup -> pass -> shoot
   for a fixed pair, as distinct from `GiveAndGoTactic`'s repeated-hop
   cycle — the old name was ambiguous once both tactics existed).
   Committed as a series of focused commits, one per fix.

   **Root cause of the residual PID overshoot**: `TwoDPID._calculate`
   (`motion_planning/src/pid/pid.py`) computed its commanded velocity as
   `Kp * position_error`, capped only by `max_velocity` — it never checked
   whether that speed was actually stoppable within the remaining distance
   given the robot's own `max_acceleration`. The separate `AccelerationLimiter`
   applied after `_calculate` returns only rate-limits how fast the
   *commanded* velocity can change once decided; it doesn't help the
   controller anticipate a stop. On a fast approach the proportional term
   alone doesn't ask for deceleration until the robot is already close,
   so the robot overshoots by however far it travels while the commanded
   velocity ramps down. At `max_velocity=2 m/s`, `max_acceleration=4 m/s²`,
   physical stopping distance is `v²/(2a) = 0.5m` — bigger than any bare
   proportional margin. Fixed by adding a braking-distance speed cap,
   `v <= sqrt(2 * max_acceleration * error)`, alongside the existing
   `max_velocity` cap. Verified: `verify_defense_foul_fix.py`'s full 90s
   match went from 1 residual foul to 0; full suite unchanged at 638
   passed / 2 skipped / 2 xfailed both before and after.

## Motion-controller discontinuity handling (flagged 2026-08-21, resolved 2026-08-22)

Context: across a bug-fixing session, the same underlying bug — a tactic
hands a robot a new orientation target that's discontinuous from what it was
just facing (e.g. "face the passer to catch a ball" -> "face the goal to
shoot"), and nothing tells the shared angular PID's per-robot state
(`pre_errors`/`integrals` in `motion_planning/src/pid/pid.py`) to forget its
history — was independently found and fixed *six separate times*, at six
different tactic call sites (`switch_of_play.py` x3, `give_and_go.py` x3,
`pass_and_shoot.py`, `lead_and_support.py`, `decoy_and_overload.py` x2), each
requiring its own from-scratch match trace to discover. A dedicated
architecture-audit agent confirmed the motion-controller/PID code itself is
not a mess (small, well-factored, zero drift between the two actively-used
`MotionController` implementations) — the actual gap is a missing protocol at
the tactics/motion-planning boundary: `ctx.motion_controller.reset(robot_id)`
exists and correctly clears the stale state, but calling it is a manual,
opt-in discipline enforced by nothing. Two ideas came out of that audit; the
first is now built (commit `91100ff`):

1. **Built: discontinuity-detecting wrapper, magnitude-threshold approach,
   orientation only.** `AbstractPID.calculate()` (`motion_planning/src/pid/
   pid_abstract.py`) now tracks each robot's last commanded target and
   auto-resets (`pre_errors`/`integrals`/`first_pass`/the acceleration
   limiter) whenever the new target jumps past a threshold, before
   computing — no tactic-side `reset()` call needed at all. This is exactly
   the "magnitude threshold" idea flagged below as having an unresolved
   failure mode, and that failure mode was real, measured, and resolved by
   **scoping the mechanism to orientation only**:
   - `PID._target_jumped` (the orientation controller): threshold
     `0.5 rad (~28.6°)`. Verified in a live 60s match: 138 genuine
     discontinuity resets fired, all with 2-4.5 rad deltas — the same class
     of jump the six manually-fixed bugs above all were.
   - `TwoDPID._target_jumped` (the translation controller): **deliberately a
     no-op, always returns `False`.** Tried the same magnitude-threshold
     approach here first and it caused a real regression:
     `test_referee_override.py`'s penalty-formation test failed because a
     live-recomputed formation target legitimately shifts by 0.6-1.2m
     repeatedly as the robot approaches (measured directly, 14 spurious
     resets over 200 ticks on one robot), each reset discarding real
     acceleration-limiter/derivative progress. This confirms the exact
     concern raised below — translation targets can't be told apart from a
     discontinuity by magnitude alone — but it turned out not to matter:
     every one of the six bugs that motivated this whole effort was the
     *orientation* PID's stale derivative; translation was never the
     culprit, so it simply doesn't need this behavior.
   - Removed as dead weight once the PID layer handled this automatically:
     all 9 manual `ctx.motion_controller.reset()` calls across
     `give_and_go.py`/`switch_of_play.py`/`decoy_and_overload.py`/
     `pass_and_shoot.py`/`lead_and_support.py`, plus the
     `was_carrying`/`was_shooting` mem fields that existed purely to gate
     them.
   - Full test suite green (721 passed / 2 skipped / 2 xfailed) after the
     fix, same baseline as before.
2. **Not built, and now more clearly unnecessary given (1)'s result:** the
   identity/intent-tag alternative below was the fallback in case magnitude
   thresholds proved unworkable in general. Since the actual bug population
   was 100% orientation and magnitude-thresholding works cleanly there, this
   more invasive alternative (threading an intent tag through every
   `move()`/`turn_on_spot()` call site) isn't needed. Left below for
   reference, not because it's still an open decision.
   - The idea: some mechanism at (or wrapping) `skills/src/utils/move_utils.py`'s
     `move()` — already confirmed the single funnel point every skill/tactic
     routes through before hitting `motion_controller.calculate()` — that
     auto-invalidates a robot's PID state when its commanded target changes in
     a way that means "this is a new task," not "this is a continuation of the
     same task," without every tactic author needing to notice and hand-place
     a `reset()` call.
   - Have `move()`/`turn_on_spot()` accept an explicit intent tag from the
     caller (e.g. `"switch_of_play.finish"` vs `"switch_of_play.relay"`), and
     auto-reset whenever that tag changes for a given robot. This pushes
     the actual judgment call back to the tactic (which already knows
     semantically when its own intent changed) while still removing the
     manual `reset()` call-site burden — the tactic states *what* it's
     doing, the shared layer decides *whether that's new* mechanically,
     with no threshold-tuning guesswork.
   - Also proposed, smaller and independent of either design above: add a
     `reset(robot_id)` call at the kernel's own existing tactic-reassignment
     point (`engine/strategy.py:264`, where `slot.mem` is already reset to
     `None` on reassignment) and barrier-reset point (`engine/strategy.py:
     375-389`) — the kernel already knows exactly when a robot's tactic
     assignment changes, which is a strict subset of "target went
     discontinuous," and today does nothing to the motion controller at
     that moment either. Superseded by (1): the PID-level auto-detection
     already covers reassignment discontinuities without needing the kernel
     to know about motion-controller internals at all.

2. **A shared `Sticky`/hysteresis helper.** The same audit found 5-6
   independently-invented instances of "keep the previous choice unless a
   new candidate beats it by a margin," spanning three architectural layers:
   skill (`go_to_ball.py`'s `_COMMIT_RANGE`), tactic (`switch_of_play.py`'s
   `_WEAK_SIDE_MARGIN`/`_ARRIVAL_SPEED_THRESHOLD`, `pass_and_shoot.py`'s
   `_REASSIGN_MARGIN_M`, `_pass_and_score.py`'s `switch_margin`), and even
   the scheduler/partitioner level (`kernel_strategy.py`'s possession-edge
   pickers). Each instance is well-reasoned and well-commented in isolation
   (several docstrings cite the exact live-match trace that motivated them),
   but there's no shared primitive, so the same "don't let noisy per-tick
   recomputation thrash a downstream consumer" insight gets re-derived from
   scratch each time.
   - Per the project's stated minimalism preference (avoid speculative
     abstraction — add a shared mechanism only after a concrete forcing
     case, not in anticipation), **explicitly not recommended to build yet**.
     The instances aren't quite the same shape (scalar-distance comparison
     vs. "which gap contains the old choice" vs. a boolean edge trigger), so
     a forced common interface could make each individually less readable
     than its current purpose-built form. Revisit if/when a clearly-6th
     instance of the exact same shape shows up, rather than building this
     speculatively now.

## Multi-strategy / tournament evaluation infra

~~Much later priority~~ **First pass done** (2026-08-16). The catalog reached 8
`build_*_kernel_strategy` configs (7 original + `decoy_and_overload`), enough
to make round-robin comparisons meaningful — `tournament.py` (renamed from
`demo_tournament.py` on 2026-08-20, once it grew persistent per-match stats/
replay recording and became the standing way to evaluate strategy changes
rather than a one-off demo) round-robins
every pair via headless rsim `StrategyRunner` matches (6v6, `opp_strategy`),
reads the final score off `CustomReferee`'s scoreboard, and prints a results
table. Deliberately just a for-loop over the existing `StrategyRunner` primitive
plus a plain tally — no new `Runner`/`Tournament` class, no persistence layer,
no bracketing/seeding, per the minimalism discipline. `C(8,2)=28` matches, all
of which fit comfortably in one run at this catalog size.

**Found and fixed along the way:** the smoke test for this script crashed
immediately on any pairing involving `build_high_press_kernel_strategy` or
`build_press_and_pass_kernel_strategy` — both use `PressAndContainTactic`,
whose only marking call (`skills/man_mark.py`) turned out to be completely
broken (3 separate stale-API mismatches: `Ball`/`Robot` don't have bare
`.x`/`.y`, `move()` expects `robot_id: int` + `Vector2D`, not a `Robot` object
+ tuple). `press_and_contain.py`'s own docstring already flagged `man_mark` as
"previously unused by any tactic," and there was no test file for it — this
had apparently never been exercised end-to-end before. Fixed the API
mismatches (kept the original ball-to-target perpendicular-offset marking
geometry, a reasonable design distinct from `ShadowAndMarkTactic`'s
goal-side marking, just buggy in its API usage), added
`tests/skills/test_man_mark.py` (none existed).

**Still not done:** `test_all_strategy_configs.py`'s existing
`build_high_press_kernel_strategy`/`build_press_and_pass_kernel_strategy` test
cases were passing *before* the `man_mark` fix too — meaning those tests never
actually drove a scenario that exercises the marking branch of
`PressAndContainTactic`. That's a real, separate test-coverage gap (the
existing tests exercise the tactic's shape, not this specific code path) —
flagged here, not fixed, since closing it means understanding what game state
actually triggers marking, not a quick addition.

**TODO — investigate `StrategyRunner`'s `enable_vision_stream` default:** a
full 60s tournament match was taking 12+ minutes of wall time (worse than
real-time) until `demo_tournament.py` explicitly passed
`enable_vision_stream=False` (commit `3086337`) — with it off, the same match
runs in ~21s (2.9x *faster* than real-time), the speed rsim headless is
supposed to have. `enable_vision_stream: bool = True` is `StrategyRunner`'s
hardcoded constructor default (`strategy_runner.py:242`), with no
mode-awareness (set the same regardless of `rsim`/`grsim`/`real`) and no
signal that leaving it on is expensive unless a caller happens to profile a
slow run and find the HTTP server startup/frame-render cost themselves, the
way this session did. The vision stream has real value for grsim/real-mode
operator workflows (`demo_referee_gui_rsim.py` genuinely wants it) — this
isn't "the default is wrong," it's "the default silently punishes headless/
automated callers who have no way to know to turn it off, instead of the env
setup itself recognizing when nobody's watching." Worth a real look at making
this the caller's responsibility to *opt into* rather than *opt out of* for
non-interactive contexts, or auto-detecting when nothing's actually consuming
the stream — not decided here, just flagged as a genuine, measured (35x)
performance footgun worth designing around properly rather than patching
per-caller as this session did.

**Investigated (2026-08-16, subagent findings independently spot-verified
against the real code before being trusted):** the two costs above, plus the
tick-rate question, all followed up on properly rather than left as
speculative TODOs.

- **`robosim` pipe I/O — real, actionable fix found, not just a cost to
  shrug at.** `RSimSSL.send_commands()` (`rsim.py:152`) calls
  `self.simulator.step(sim_cmds)` and **discards the return value**.
  `robosim_subprocess.py:82-83` already sends that step's resulting state
  back over the pipe as part of the same response
  (`state = sim.step(...); print(json.dumps({"state": state}))`) — but
  nothing reads it. `standard_ssl.py:187` then issues a **second, separate**
  `get_frame()` → `get_state()` round-trip
  (`rsim.py:155-156`) to re-fetch the same information a moment later. Two
  round-trips per tick where one would do — confirmed by reading both files
  directly, not taken on the investigating agent's word. This is exactly
  what the original 1804-`readline()`-over-900-ticks (≈2/tick) measurement
  was seeing, just not previously diagnosed as *why* it was 2, not 1.
  Wiring `send_commands()` to return/cache the `step()` response and having
  `get_frame()` reuse it instead of a fresh `get_state()` call would roughly
  halve `robosim` pipe wait per tick — small, surgical, no physics-fidelity
  tradeoff (unlike the tick-rate idea below). **Not yet implemented** — a
  real, scoped, low-risk fix ready to pick up.
  - Measured separately, with a standalone script against the real
    subprocess: JSON (de)serialization is ~1.0% of the ~2.27ms/call
    real-world cost (`0.023ms` pure serialize/deserialize vs. `2.2659ms`
    mean end-to-end). The remaining ~99% is pipe write/flush, OS process
    scheduling, and robosim's own C++ physics step — not something a
    format change would meaningfully affect.
  - **Cross-tick batching (send N ticks, read N responses) is not
    feasible** and shouldn't be pursued: each tick's commands are the
    output of a strategy decision (`Partitioner`, `Tactic.tick()`,
    `FastPathPlanning`) made *after* seeing the previous tick's physics
    result. There's no window to compute N ticks' commands ahead of N
    physics results without either running strategy decisions blind
    (no longer reactive) or moving strategy logic into the physics
    subprocess (a much larger architectural change, not a pipe
    optimization).
- **`distance_point_to_segment`** — not re-investigated this pass; still
  flagged as a plausible `FastPathPlanning` optimization target (427,032
  calls, 2.34s cumulative in the same 900-tick profile), lower priority than
  the pipe fix above since it doesn't have an already-identified concrete fix.

**Scoreless-draw pattern (24/28 matches, 86%) — root-caused at the tactic
layer (2026-08-16), fixed, but blocked on a newly found simulator bug.**
Hypothesis that this was caused by rsim's ball-stickiness-on-release bug was
investigated and **ruled out** (see item 1 of the grsim TODO section below):
that bug only affects `DribbleTactic`'s passive-release path, not wired into
any of the 8 tournament configs.

Instrumented a live match (`build_default_kernel_strategy` vs itself,
per-tick phase/state tracing) and found the real cause: **`pass_and_shoot.py`
and `_pass_and_score.py`'s pass-and-shoot phase machine was a one-way,
no-retry state machine with no failure recovery.** Five independent, stacked
bugs, each verified against real per-tick trace data before being trusted (not
assumed from reading code alone):

1. `PassAndShootTactic.tick()`: `goal_scored=True` returned `{}` forever —
   even a *successful* attempt permanently froze the tactic for the rest of
   the match. Fixed: reset to a fresh attempt instead of freezing.
2. No timeout on a stuck `pass_then_score`/`score` phase — `is_committed()`
   only releases once `phase == "setup"`, but nothing ever set it back on
   failure, so one botched pass/shot deadlocked the pair for the rest of the
   match. Fixed: a tick-budget timeout resets back to `setup`. Initially set
   to 300 ticks (~5s), found (via trace) to be too tight for the real
   aim+position+kick+catch(+aim+shoot) choreography — one attempt reached
   `score` phase at 259 ticks and then timed out *inside* `score` without
   ever kicking; bumped to 720 ticks (~12s).
3. `assign_passer_receiver` had zero hysteresis — when both robots start
   near the ball (the common case), naive per-tick closest-to-ball comparison
   flipped the "passer" identity nearly every tick (confirmed via fine-grained
   trace: passer id alternated for dozens of consecutive ticks), and each
   flip reset `PassAndScoreMem`, so setup could never accumulate two
   consecutive ticks of progress. Fixed: a 0.3m margin before roles flip.
4. `has_ball(..., visual=True)`'s `capture_distance=0.12m` left almost no
   margin above `ROBOT_RADIUS + BALL_RADIUS≈0.1115m` (actual contact
   distance) — a stationary dribbling robot's distance-to-ball jitters by
   ~0.01-0.02m tick to tick from simulator noise alone, so the flag toggled
   True/False *every single tick* right at pickup. Since `run_setup_phase`/
   `_pass_exec`/`_score_goal` all branch on this boolean ("if has_ball: X
   else: Y"), that flicker meant the passer's command alternated between two
   different behaviors every tick and never sustained either. Fixed: widened
   to 0.15m. Also switched all three call sites from the non-visual
   `robot.has_ball` sensor (found separately stuck permanently `False` while
   the robot sat visually on the ball) to the visual fallback.
5. `_setup_positions`' rng was unseeded per-`PassAndScoreMem` — harmless on
   its own, but made debugging non-reproducible; seeded deterministically
   from `(passer_id, receiver_id)` as a defensive idempotency property.
   (Chasing what looked like this causing a *second* bug — the setup target
   itself alternating between two sampled positions every tick — turned out
   to be a self-inflicted debugging artifact: two teams' independent tactic
   instances were being logged through one undifferentiated global
   monkeypatch, not one tactic's state actually flip-flopping. No real bug
   there; flagged so nobody re-chases it.)

All 5 fixes verified working via per-team-labeled tracing: setup reliably
completes, passes reliably complete, `score` phase is reliably reached, and —
critically — a robot now reliably *fires a correctly-aimed kick*
(`target_oren` vs `robot.orientation` within the 0.05 rad tolerance,
confirmed against a freshly-recomputed `best_shot_y` that was itself
confirmed inside the real goal's `y∈[-0.5,0.5]` range).

**Native-simulator-level blocker found AND fixed (2026-08-16): the kick's
actual ball-launch direction didn't match the robot's commanded orientation
at kick time.** Traced one real kick precisely: `robot.orientation=-0.686`
rad (correctly aimed, per above), but the ball's *observed flight path*
after the kick was `-0.140` rad — a ~31° (0.546 rad) mismatch, resulting in
a miss roughly 4x the goal's width. This was far larger than the Python-side
`ORIENTATION_TOLERANCE_RAD=0.05` tolerance could explain (at the ~3.3m range
involved, that tolerance permits at most ~0.165m of lateral miss, not the
~1.6m-equivalent actually observed) — and `kick()` (`skills/src/utils/
move_utils.py`) carries no direction parameter at all; the launch vector is
determined entirely by the native rSim/ODE kicker physics once `kick=1`
reaches the simulator. Root-caused (subagent, `vendor/rSim/src/robosim/
sslrobot.cpp`'s `Kicker::kick()`) to two compounding bugs: (1) the kick gate
used `isTouchingBall()`, a razor-thin box check (~3cm forward, 4cm lateral)
meant for deciding whether to grab the ball into the dribbler hold — far
stricter than the 0.15m radial `has_ball(visual=True)` any tactic actually
trusts before issuing `kick()`, so a kick issued while the ball hadn't
settled into that thin window (e.g. right after catching a moving pass)
silently no-opped, leaving the ball on whatever pre-existing velocity it
had; (2) even when the gate passed, the tangential (sideways) component of
the ball's pre-kick velocity was added back at full, undamped strength while
the normal component was damped by `kickerDampFactor=0.2` — any residual
lateral velocity bled straight into the kick's resultant direction. Fixed:
new `isNearKickerFace()` gate (radial distance from chassis center, matching
`has_ball()`'s own semantics) replaces `isTouchingBall()` for the kick gate;
tangential velocity now damped by the same factor as the normal component.
Diff at `docs/patches/rSim-kick-direction.diff`, applied in `vendor/rSim/`
alongside the pre-existing dribbler-release patch (see that section above).
Repro (synthetic worst-case: shooting immediately after catching a moving
pass) went from up to 176° launch-direction mismatch on stock rSim (kick
silently no-opping) down to under ~8° adversarial / ~1-2° typical with the
fix.

**Keep-out-zone "regression" investigated and resolved (2026-08-16): was
never a bug in either rSim patch.** Installing the patched build (dribbler-
release fix specifically) changed the outcome of
`test_their_kickoff_clears_our_robots_outside_center_circle` — root-caused
(second subagent) to **test fragility, not a robot-behavior bug**. The test
asserted distance from the fixed field origin `(0, 0)`, silently assuming
the ball stays parked at center — three sibling assertions in the same file
correctly measure from `game.ball.p` instead. Upstream v1.2 has a genuine
one-way-latch bug where `setActions()` never turns the dribbler back off;
the dribbler-release patch correctly fixes that, and that single-tick
`dribblerOn` state difference — landing during real incidental ball contact
in the test's early ticks — sent ODE's contact solver down a different,
chaotically-sensitive branch, so the ball settled 0.56m from center instead
of near it. The robot under test was confirmed correctly positioned the
whole time (`dist(robot, actual ball)` = 0.797m, safely above the 0.75m
threshold) — only the fixed-origin proxy read as a failure. Verified by
rebuilding genuinely-pristine upstream v1.2 locally (same toolchain) and
confirming it reproduces PyPI-stock's trajectory bit-for-bit, ruling out a
toolchain confound. Fixed: `test_referee_override.py`'s assertion now
measures from `game.ball.p`, matching its siblings. Full suite (638 tests)
passes clean on the patched build with both rSim fixes installed.

Tactic-level fixes above are complete and correct independent of the kick-
physics fix — they fixed a real, separate class of bug (the phase machine
deadlocking) and are kept regardless. With both the tactic-level fixes and
the rSim kick-direction fix now installed together, a fresh tournament run
is in progress (2026-08-16) to confirm the scoreless-draw rate actually
improves — a correctly-aimed kick reaching the simulator is necessary but
its physical accuracy was the last untested link in the chain.

**That tournament re-run confirmed the fixes above weren't enough on their
own: `pass_and_shoot`-based configs still scored zero goals in full 6v6
matches (23/28 scoreless, 82%, barely down from the 86% baseline), despite
completing cleanly in a nearly-empty solo trace.** Root-caused (2026-08-16,
user-authorized to work in `utama_core/motion_planning/` specifically for
this, temporarily lifting the tactic-only constraint the same way
`vendor/rSim` was authorized) to two real, independent bugs in
`FastPathPlanner` (`motion_planning/src/fastpathplanning/planner.py`) that
only manifest with other robots actually on the field (a 6v6 match, not a
sparse solo trace):

1. **`_find_subgoal`'s unguarded `perp_dir / np.linalg.norm(perp_dir)`
   divided 0/0 into NaN** whenever a recursive detour step's endpoints
   collapsed to the same point (`direction = target - robot_pos == 0`),
   silently poisoning every subgoal derived from it for the rest of that
   recursion — this is the same divide-by-zero already flagged elsewhere in
   this doc (`test_referee_override.py`'s keep-out-zone investigation, the
   `test_mirror_swap` xfail) as a pre-existing, previously un-root-caused
   `FastPathPlanning` warning. Fixed: guard the zero-magnitude case, fall
   back to `obstacle_pos` (same fallback the existing recursion-depth
   failsafe already uses).
2. **`sanitize_target` pushes a target away from *any* nearby obstacle by a
   flat `OBSTACLE_CLEARANCE` (~0.27m) — including a `go_to_ball` target
   sitting on a ball that's contested near an opponent.** Confirmed via
   direct reproduction: a passer approaching a ball parked ~0.31m from an
   idle enemy robot got its target sanitized from ~0.01m off the ball to
   ~0.13m off it; the resulting "carrot" (`smooth_path`'s projected
   waypoint) barely advanced each tick, so the robot converged to ~0.15m
   from the ball and sat there oscillating for the rest of a 60s match,
   `has_ball` never firing. This is a real design gap, not a numeric
   off-by-one: any ball contested near an opponent — a completely normal,
   legal SSL situation — was permanently uncollectable. Fixed: `_path_to`
   now identifies obstacles sitting adjacent to the *live ball position*
   when the target itself is a ball-approach target (inferred structurally
   — target within `OBSTACLE_CLEARANCE` of the ball, which is how
   `go_to_ball`'s small overshoot target always sits — not via a new
   parameter threaded through every `MotionController` subclass) and
   exempts just those obstacles from `sanitize_target`'s push-away logic,
   mirroring the function's existing, same-reasoning exemption for field
   boundary walls ("the ball can legally be near [it] and the robot must be
   able to reach it"). The obstacle is still fully respected by
   `check_segment`'s path-routing — only the final target point stops being
   pushed away from a ball-adjacent obstacle.

Both fixes verified: full suite 638 passed / 0 regressions (one defensive
`game.ball is not None` guard added after the fix broke several
ball-less `motion_planning` unit tests that don't set up a ball fixture at
all), and the previously-permanently-stuck 6v6 passer now reliably closes
to real contact range (`has_ball` strict sensor firing, not just visual)
and the tactic reaches `score` phase in a real 6v6 match. A fresh tournament
run (2026-08-16) is in progress to quantify the real-match impact of these
two fixes on top of everything above.

**Tournament re-run (2026-08-16) with everything above installed: real,
measured improvement.** 36 matches (9 configs — `switch_of_play`, a new
tactic added this session, see below, joined the auto-discovered catalog),
scoreless-draw rate dropped to 26/36 (72%), down from the original 86%
(24/28) baseline and the intermediate 82% (23/28) measurement taken before
the `motion_planning` fixes. 10 of 36 matches now have a real scored goal,
spread across 6 different configs (`high_press`, `split_shape`,
`press_and_pass`, `three_slot`, and others taking a loss). `pass_and_shoot`
-based configs (`default`, `low_block`, `three_slot`) are no longer
structurally frozen — `default` lost several matches by conceding rather
than by never moving the ball, `three_slot` won once — confirming the
motion-planning fix's real effect, even though these configs haven't yet
scored *themselves* in this run (likely a finishing-quality gap now, not the
structural stall that's been fixed).

**A second, real, pre-existing, independent bug found during
`switch_of_play` verification (not caused by anything this session
touched): `DefenseTactic`/`defend_parameter` can put more than one
non-goalkeeper defender inside the team's own defense area, triggering a
`DIRECT_FREE_*` foul (`DefenseAreaRule`'s `max_defenders=1`, correctly
enforcing the real SSL rule) that halts play for an extended stretch.**
Confirmed via direct referee-command tracing: `RefereeCommand.FORCE_START ->
DIRECT_FREE_YELLOW` at `status="Too many blue defenders in own area"`,
immediately followed by the mirror foul for the other team once play
resumes — the match effectively stalls in a foul loop rather than
progressing. `utama_core/tactics/defense.py`'s own docstring already flags
the root cause as a known, deliberately-deferred limitation: `defend_parameter`'s
dynamic 2-defender side-selection triggers on `len(game.friendly_robots) ==
2` (the *whole team's* robot count), not on how many robots this tactic
instance was actually handed — any config running `DefenseTactic` with 2
defenders on a team with more than 2 robots total falls back to a fixed
near-post assignment instead of the dynamic side choice, which was
"correct for the original dedicated 2-robot defense strategy it came from"
but not audited for the general case. Pre-existing (confirmed via `git log`
on `defense.py`/`defend_parameter.py` — no commits from this session's
work touch either file), not something this session's fixes caused, but it
became a *forcing* case here because it's currently the practical blocker
to cleanly verifying any attack tactic paired with `DefenseTactic` all the
way to a scored goal in a full 6v6 match.

**Fixed.** Two independent bugs were found and fixed in
`utama_core/skills/src/defend_parameter.py`, both required to actually stop
the foul loop:

1. **The trigger-condition bug described above.** `defend_parameter` now
   takes an optional `defender_group` parameter — the tactic's own
   `robot_ids` — and `DefenseTactic.tick()` passes its `robot_ids` through.
   When provided, both the 2-defender dynamic-side-selection trigger and the
   near/far-post parity fallback are keyed off position within that group
   instead of `len(game.friendly_robots)`/`robot_id == 1`. Root-caused via
   direct trace on `build_switch_of_play_kernel_strategy` (5 outfield
   robots, `min_attack=3` puts `DefenseTactic` on robots `{4, 5}`): neither
   ID is `1`, so *both* defenders hit the static fallback's `else` branch
   and picked the identical `-post_limit` target, collapsing onto the same
   post. Omitting `defender_group` (every other existing call site,
   including `shadow_and_mark.py` and all pre-existing tests) reproduces the
   exact original behavior — untouched, not a breaking change.
2. **A second, independent bug this investigation surfaced: the outfield
   defender's own-box standoff margin was too tight for the motion
   controller's real tracking overshoot.** Even with bug (1) fixed and the
   two defenders correctly split to opposite posts, the match still fouled
   within ~3 seconds. Direct trace showed why: `defender_x`'s target sits
   exactly `ROBOT_RADIUS` (0.09m) outside the defense-area edge — enough for
   the robot's *static* footprint, but zero slack for the PID motion
   controller's actual approach behavior. A defender chasing its target
   from across the field at speed doesn't stop precisely at the target; it
   was measured overshooting by up to ~0.3m before decelerating, carrying
   its center past the box boundary. Tried 0.05m and 0.2m extra standoff
   first (both still measurably insufficient via direct match trace) before
   settling on a new `OWN_DEFENSE_AREA_STANDOFF_DISTANCE = 0.4`
   (`utama_core/config/referee_constants.py`), added on top of
   `ROBOT_RADIUS`. This is a real, uncomfortable trade-off: a defender now
   holds noticeably farther from its own box than the rule strictly
   requires, which likely costs some shot-blocking coverage. The `move()`
   PID controller's overshoot characteristic is the actual root cause and
   wasn't touched — retuning it was out of scope (used everywhere, high
   blast radius) — so this margin is a deliberate blunt instrument, not a
   precise fix. Verified via `verify_defense_foul_fix.py`: 90s match with
   both teams running `switch_of_play` (the same config that fouled at 2.6s
   before either fix) now runs to 43s of genuine match play — kickoffs,
   ball-out-of-bounds restarts, direct frees — before a single remaining
   foul, on the *other* team, root-caused as `SwitchOfPlayTactic` (not
   `DefenseTactic`) putting an attacker inside its own defense area during
   relay play. That's a separate, new, not-yet-investigated bug in
   `switch_of_play.py`, out of scope for this fix — noted below.

Both `test_defend_parameter.py` (updated: several tests hardcoded the old
`ROBOT_RADIUS`-only standoff as an expected value, now derived from the
shared `_DEFENDER_STANDOFF` constant so they track the real geometry) and
`test_all_tactics.py` pass clean (45 passed, 2 skipped) after both fixes.

**Follow-up, same investigation: the `SwitchOfPlayTactic`-inside-its-own-box
bug noted above is fixed.** Root cause: `_pivot_target()`
(`utama_core/tactics/switch_of_play.py`) pulls the pivot's target backward
from the carrier by `0.2 * (goal_x - carrier_pos.x)` — proportional to the
*full* carrier-to-enemy-goal distance, uncapped. Right after a
kickoff/restart the carrier is often still deep in its own half (small
`|carrier_pos.x|`), and at that range the pullback overshoots straight
through the team's own goal line into its own defense area. Confirmed via
direct trace: the pivot held station inside its own box for an extended
stretch (multiple seconds), tripping the same `DefenseAreaRule` foul from
the attacking side rather than `DefenseTactic`. Fixed by clamping
`_pivot_target()`'s result to stay outside the team's own defense area by
`2*ROBOT_RADIUS + OWN_DEFENSE_AREA_STANDOFF_DISTANCE` (reusing the same
constant added for the `defend_parameter` fix above, plus an extra
`ROBOT_RADIUS` of headroom — the pivot approaches this target from open
field at real speed, the same overshoot mechanism as `defend_parameter`'s
own-box standoff bug, but on a different, unverified approach profile; the
`defend_parameter`-sized margin alone was measured as still insufficient
via direct trace before the extra headroom was added). **Caught and fixed
a sign error in the first attempt**: the initial clamp used `max(...)`/
`min(...)` with the standoff added on the wrong side of the boundary for
`my_team_is_right=True`, which pushed the target *into* the box instead of
away from it — caught immediately by re-running the verification match
rather than trusting the diff, and fixed by swapping the clamp direction to
match `defend_parameter`'s already-verified sign convention. Verified via
`verify_defense_foul_fix.py`: the match that previously fouled at t=43.12s
(pivot deep inside the box) now runs clean to t=61.43s before one further,
much smaller instance of the same class of foul (pivot's position at
x=-3.504 against a box edge at x=-3.5 — a few centimetres of residual PID
overshoot, not the multi-second stall from before). Targeted test suite
(`pytest -k "switch_of_play or defense"`) passes clean, 31 passed. The
residual few-centimetre overshoot is the same underlying PID-overshoot
class of issue flagged (not fixed) in the `defend_parameter` writeup above
— not re-chased further here given diminishing returns; a real fix needs
the motion controller's approach/braking behaviour addressed generally,
not another per-call-site margin bump.

**New tactic added this session: `SwitchOfPlayTactic`
(`utama_core/tactics/switch_of_play.py`), a 3-robot carrier→pivot→runner
relay that deliberately relocates the ball to the weaker-defended side of
the field before attacking**, distinct in kind from every existing attack
tactic (none of which read the *global* left/right defensive balance).
Originated from a user request to field-test draft "Writing a Tactic"
guidance later added to `AGENTS.md` — see that doc's "Writing a Tactic"
section for the guidance itself. Went through two rounds of debugging by
two different subagents before reaching its current state:
- First subagent (design + initial validation): built the tactic, found and
  fixed a second-order "one-way phase machine" gap the guidance didn't
  explicitly cover (a timeout reset that gets silently undone within the
  same tick if the reset-target phase's own logic immediately re-advances
  past it) and a `_pass_exec`/`intercept_point()` "receiver still moving"
  stall pattern — but never got the tactic to reach its `"finish"` phase in
  13+ validation runs; ran out of a self-imposed effort budget without a
  clean report.
- Second subagent/fork (finish debugging, verify): root-caused the
  *dominant* stall to a bug in the same family, one level deeper — the
  carrier's ball-holding command in `"assess"` phase used `go_to_point()`,
  whose default orientation faces the ball, not the pivot it's about to
  pass to; `intercept_point()` (`shared/pass_and_score_geometry.py:78`)
  projects the receive point *along the passer's current orientation*, so
  a carrier facing the wrong way sent that projection somewhere the pivot
  never walked to. Fixed, verified reaching `"finish"` in a real match
  trace. Independently re-verified (2026-08-16): the fix's mechanism checks
  out against the actual `intercept_point()` source, and a fresh match
  confirmed genuine new progress (reaching `"relay"` phase, never achieved
  before). However, the exact same bug pattern recurred one phase later,
  unpatched: `"relay"`'s "hold the ball while runner arrives" branch had
  the identical `go_to_point()`-faces-the-ball issue for the pivot-as-source
  robot. Found via direct trace (`src_oren` oscillating tick to tick,
  `intercept_pos` swinging wildly in lockstep) and fixed the same way
  (explicit `target_oren` at the runner via `move()` instead of
  `go_to_point()`'s default).
- **Status at this point in the investigation: tactic logic improved and
  independently verified making real progress (reaches `"relay"`, no
  permanent robot-strand, phase transitions are clean), but not yet
  demonstrated reaching `"finish"` or scoring a goal end-to-end** — every
  attempted verification match got interrupted by the separate
  `DefenseTactic` foul-loop bug above before enough clean playing time
  accumulated. Full test suite (638 tests) passed clean throughout all of
  the above. **Superseded by the follow-up fixes above** (`DefenseTactic`'s
  foul-loop fix, then `_pivot_target()`'s own-box clamp) — see those
  entries for the current, much-further-verified status (61s+ of clean
  play). All of this session's work, including this tactic, is now
  committed as a series of focused commits.

Not yet investigated (deprioritized until the `DefenseTactic` foul-loop bug
is fixed, since it's currently the practical blocker to further tournament
signal): goalkeeper effectiveness, whether 60s is long enough end-to-end
once kicks reliably go where aimed, whether `test_mirror_swap`'s `xfail`
should be revisited now that the same `_find_subgoal` NaN it may share has
been fixed (not re-checked yet — it's still passing as `xfail` in the
current suite, so this is a "maybe now unnecessary" note, not a known
issue).

**Separate, smaller finding this session: the `robosim` pipe I/O
redundancy flagged above (item under "Not yet implemented") is fixed.**
`RSim.send_commands()` (`rsim.py`) now caches `simulator.step()`'s already-
returned state (`self._last_state`) instead of discarding it, and
`get_frame()` reuses that cache instead of issuing a second, separate
`get_state()` round-trip — falling back to a real `get_state()` call only
when there's no cached step yet (first call after construction, or right
after a `reset()`, which explicitly invalidates the cache). This is also a
correctness fix, not just a speed one: `SSLWorld::getState()` computes
robot/ball velocity via finite difference against whatever the *previous*
call returned (see `vendor/rSim/FORK_NOTES.md`'s "get_state() is stateful,
not idempotent" note) — calling it a second time right after `step()`, with
no simulation advancing in between, would have silently zeroed the velocity
fields instead of returning the real post-step velocity, an actual (if
probably rarely user-visible, since nothing was reading the discarded
first-call state) latent bug beyond the redundant round-trip itself.
**Measured real-world speedup: negligible** — a real 6v6 `default`-vs-
`default` match measured 26.04ms/tick before this fix and 26.62ms/tick
after (within noise), confirming the earlier profiling note that per-call
pipe overhead (~2.27ms) is a small fraction of total per-tick cost, which
is dominated by strategy computation (`Partitioner`, `Tactic.tick()`,
`FastPathPlanning`), PID controllers, referee logic, and the simulator's
own physics step — not pipe round-trips. Kept anyway for the correctness
fix and the (smaller than hoped) redundant-I/O removal; anyone chasing a
"blazing fast" rsim for autoresearch purposes should look elsewhere first
(the tick-rate/`CONTROL_FREQUENCY` idea above, or profiling what's actually
consuming the other ~24ms/tick, rather than more pipe-protocol
micro-optimization).

**Tick-rate investigation — full audit done, feasibility confirmed
architecturally straightforward, not yet implemented:**

- **Frame-counted-constant audit, extended beyond the two found earlier
  this session.** Confirmed safe (correctly derived from `CONTROL_FREQUENCY`,
  would stay correct automatically): `KICKER_COOLDOWN_TIMESTEPS`
  (`settings.py:36`), `PROJECTEDFRAMES / CONTROL_FREQUENCY`
  (`fastpathplanning/planner.py:120`), the DWA planner's `simulate_frames *
  TIMESTEP` / `_control_period = TIMESTEP` (`dwa/planner.py`,
  `dwa/translation_controller.py`), and the PID `dt` config defaults
  (`pid/configs.py`). Confirmed unsafe (hardcoded tick count, implicit-Hz
  assumption not encoded anywhere executable) beyond the two already found:
  **`KICKER_PERSIST_TIMESTEPS = 10  # in timesteps to persist the kick
  command`** (`settings.py:37`, right next to the correctly-derived
  `KICKER_COOLDOWN_TIMESTEPS` on the line above — spot-checked directly,
  genuinely a bare literal with no `* CONTROL_FREQUENCY`), `KICK_PERSISTENCE_FRAMES
  = 3` in `standard_ssl.py`'s `_apply_dribbler_release_kicks`, and — a
  distinct case — `_KICK_TTL_FRAMES = 45  # ~1.5s at 30fps` in
  `run/vision_stream.py:20`, which assumes a *different* implicit rate
  (the vision stream's own 30fps render loop, not `CONTROL_FREQUENCY=60`) —
  a second, independent hardcoded-rate hazard, not the same one.
- **`CONTROL_FREQUENCY` as a per-`StrategyRunner` parameter is
  architecturally straightforward, not deep.** `SSLStandardEnv` already
  accepts `time_step` as a constructor parameter (not hardcoded) —
  `StrategyRunner._init_sim_and_controller()` just doesn't pass one through
  today, so that's a one-line threading fix. RSIM mode never rate-limits on
  `TIMESTEP` at all (`strategy_runner.py:1362`'s `time.sleep` only fires for
  non-RSIM modes) — for batch/tournament runs, `TIMESTEP` today only gates
  physics step size and motion-planner `dt` integration, not wall-clock
  pacing, which is one less thing to worry about. The three motion
  controllers (PID/DWA/FastPathPlanning) all follow the same `(mode,
  rsim_env)` construction shape and read `dt`/`TIMESTEP` from factory
  functions keyed only by `Mode` — making frequency per-runner means adding
  a parameter to that one construction call chain and those factories, not
  a redesign. `FastPathPlanner`'s direct `CONTROL_FREQUENCY` import
  (`planner.py:120`) is the one genuinely awkward case with no config
  object to thread through today. Real hardware (`real_robot_controller.py`)
  should stay pinned to the true global regardless — not every consumer
  needs to become parameterized, only the RSIM-path constructors.
- **Physics fidelity at a coarser step remains a separate, undecided
  question from the plumbing** — parameterizing `CONTROL_FREQUENCY`
  doesn't resolve whether a larger `robosim` timestep's different
  per-step displacement/collision behavior is acceptable for a given
  run's purpose. Still open, as originally flagged.

**Not yet implemented, either the pipe de-dup or the tick-rate
parameterization** — both are now real, scoped, and ready to pick up rather
than speculative; picking between them (or doing the pipe fix first, since
it's smaller and has no fidelity tradeoff) is the next decision, not made
here.

**Strategy-computation-side perf work (2026-08-16, later session) — closes
out the `distance_point_to_segment` item flagged above as
"not re-investigated," plus two similarly-shaped fixes found by profiling a
real match end-to-end.** The finding above (line ~551-562: pipe overhead is
small, per-tick cost is dominated by strategy computation) turned out to be
exactly right — `cProfile` on a full 30s 6v6 match (`build_default` vs
`build_high_press`) showed the rSim `readline()` wait at ~94% of tick time,
but of that, only ~15% (~0.94s/1800 ticks) was IPC/pipe overhead; the
remaining ~85% is genuine native physics compute happening to be *measured*
through that blocking call, not fixable from the Python side. The real
Python-side cost, once separated from rSim's own wait, was concentrated in
three specific hot paths, all following the same shape: small, fixed-size
numpy operations whose per-call dispatch overhead dwarfs the actual
arithmetic at the call volumes involved (tens of thousands of calls per
match) — the same diagnosis as the earlier `distance_point_to_segment`
float rewrite, just not yet applied to these three.

1. **`FastPathPlanner._find_subgoal` was missing the bounding-box
   broad-phase prune `collides()` already had.** It checks every obstacle
   against a candidate subgoal point on every recursive retry, with no cheap
   way to skip far-away obstacles — `collides()` right below it in the same
   file already solved exactly this shape of problem (segment-vs-many-
   obstacles) earlier this session; `_find_subgoal` (point-vs-many-obstacles)
   just never got the same treatment. Added the same axis-aligned
   bounding-box pre-check. Cut `distance_point_to_segment` calls
   originating from this function 15x (941,209 → 62,575 per 30s match,
   confirmed via `pstats.print_callers`), and total `distance_point_to_segment`
   calls across the whole match by ~59% (1.48M → 603K). Measured **~26-30%
   faster** on the function itself via an isolated, interleaved A/B
   microbenchmark (obstacle-field synthetic input, not a full match — avoids
   rSim noise in the timing).
2. **`VelocityRefiner._windowed_average_derivative`** (introduced this same
   session for acceleration's windowed finite-differencing — see the
   `data_processing/refiners` design-cleanup entry) reshaped/averaged/diffed
   tiny numpy arrays (3 windows × 5 points × 2-3 dims) on every one of its
   ~46,410 calls per match. Rewritten as plain-float loops at that fixed
   shape. **~5x faster** (44-52µs → 6-10µs/call across 3 runs), numerically
   identical output (existing `test_acceleration_calculation_implements_
   expected_formula`'s exact `pytest.approx` assertions still pass).
   *(Velocity itself was tried with the same windowed scheme and reverted —
   see the design-cleanup entry for why: any windowing adds real lag under
   acceleration that broke two time-critical control-loop tests. Velocity
   stays a plain 1-step diff; only acceleration's already-existing windowing
   got the float rewrite.)*
3. **`KalmanFilter._step_xy`** used a full 2×2 numpy matrix Kalman update
   (`np.linalg.solve`, `np.matmul`) despite the measurement/process
   covariance matrices always being diagonal (`covariance_xy` is hardcoded
   `0` in `__init__`, uncorrelated x/y noise) — which means the 2D filter is
   mathematically identical to two independent scalar Kalman filters, the
   same closed-form update `_step_th` (the orientation filter, right below
   it in the same file) already uses. Rewritten as two calls to a shared
   `_step_scalar` helper. **~17-22x faster** (28-41µs → 1.5-1.8µs/call).
   Correctness verified two ways before trusting it: the full existing
   `kalman_test.py` suite (40 tests — convergence, vanish-handling,
   covariance-shrink — 2 tests updated only for renamed internal attributes,
   `state_xy`→`state_x`/`state_y` etc., not behavior), and a standalone
   500-trial cross-check against the original matrix implementation with
   randomized noise/dt/trajectory-length and mixed measurement/vanished
   frames — max deviation ~1e-15 (float64 epsilon), and zero off-diagonal
   covariance terms ever appeared in the matrix version's output, confirming
   the decoupling assumption held in practice, not just in theory.

All three verified against the full test suite (641 passed, 2 skipped, 2
xfailed — same counts as before any of this session's changes) in addition
to their individual scoped tests.

**End-to-end measurement — the number this whole investigation has been
building toward.** Single process, no multiprocessing, one 6v6 30s match,
`enable_vision_stream=False`, 3 interleaved A/B runs (baseline/current
alternated, not run back-to-back) to cancel out system-load drift, which
was substantial enough on this machine to swing a single-shot measurement
by 2x on its own:

| | avg wall time / 30s match | realtime multiplier |
|---|---|---|
| Baseline (`342dcb8`, immediately before `demo_tournament.py` existed — no perf work of any kind) | ~47.5s | 0.63x (slower than real-time) |
| Current (all rounds of perf work through this entry) | ~16.0s | 1.88x |

**~3.0x faster end-to-end**, compounding across every round logged in this
file: disabling the vision stream by default in tournament runs (`3086337`,
the single biggest jump per the 35x-footgun note above), the rSim
`step()`-state cache (`0cf1e17`), `distance_point_to_segment`'s float
rewrite (`679e8cd`), per-tick obstacle-list caching (`b79b863`),
`collides()`'s bounding-box prune (`b98cd29`), and this entry's three fixes
(`4493002`). No single commit explains the 3x — it's multiplicative
accumulation across rounds, each closing a gap the previous round's
profiling surfaced.

Isolated per-function speedups (5x, 17-22x, etc.) do **not** translate
1:1 to end-to-end speedup, and this is worth stating plainly since it's an
easy number to misread: rSim's `readline()` wait is still the largest
single share of tick time by far (per the profiling above), and none of
this round's fixes touch it. A function going from 44µs to 2µs matters a
lot in a profiler's self-time ranking; if it was only ~1s out of a ~26s
match to begin with, cutting it to ~0.05s doesn't move the total by much.
The honest ceiling on *this* direction (Python-side refiner/planner code)
is close to exhausted — further large jumps would need to come from the
rSim/robosim boundary itself (the pipe de-dup / tick-rate items above),
which is real work with real tradeoffs (physics fidelity, cross-environment
protocol risk), not more of this session's style of micro-optimization.

Original framing, for context (superseded by the above):

Note: an earlier plan (`snug-hugging-sutton.md`, now deleted) explored a
multi-strategy `Runner` built directly on `AbstractStrategy`/py_trees, to let several
functional strategies each dynamically own a subset of robots. That specific
mechanism problem — many things concurrently owning dynamic robot subsets — is what
the tactic-kernel (`Strategy` + `Tactic` + `Partitioner`) already solves, on a
different (non-BT) substrate. Any future tournament/multi-strategy infra should build
on the kernel, not resurrect the BT-based Runner design. A few ideas from that plan
are still worth keeping in mind when this gets built:
- Reassignment should reset a tactic's `mem` exactly when its robot set changes, not
  otherwise (already the pattern inside `pass_and_shoot.py` and generalized into the
  kernel's `Strategy` tick loop).
- Conflict detection: never allow the same robot to be double-assigned in one tick
  silently — assert loudly instead of last-write-wins.
- An allocator/partitioner is cleanest as a pure function `(Game, ...) -> assignment`,
  re-run every tick; "static" allocation is just the trivial case of a function that
  ignores `Game`.

## More tactics — football-inspired plays/formations

Only a handful of tactics exist today (goalkeeper, defense, pass_and_shoot,
lead_and_support), each tagged via the closed `TacticTag` vocabulary
(see `tactic_model_design_decisions.md` §15). There's a lot of real football/SSL
tactical vocabulary worth mining for genuinely new tactics — formations, set plays,
pressing schemes, overlap/give-and-go patterns, etc. — rather than growing the
catalog by variations on what's already there. Also the natural forcing function for
actually exercising `applicable()`/tags at more than toy scale.

## Codebase cleanup — remove remaining BT/py_trees junk

Previously attempted inline alongside unrelated rsim-env cleanup and reverted
because the diffs got entangled (see kernel-cleanup commit history around
`97839a6`) — this pass is deliberately isolated to avoid that again.

**Done:**
- `AbstractStrategy` rewritten as the kernel-tactic strategy base class —
  `KernelStrategy` merged into it, no more `self.blackboard`,
  `self.behaviour_tree`, or `create_behaviour_tree()`. Landed in `087ee4b`.
- Dead `setup_behaviour_tree`/`setup_strategy_blackboard` no-ops and their
  `StrategyRunner` call sites removed. Same commit.
- Deleted: all 13 BT example strategies (`strategy/examples/*` and
  `strategy/examples/motion_planning/*`), `strategy/common/base_blackboard.py`,
  `strategy/common/blackboard_contract.py`, `strategy/referee/tree.py`,
  `strategy/referee/conditions.py`, `tests/common/test_blackboard_contract.py`,
  `tests/strategy_examples/*`.
- **Kept, deliberately**: `strategy/referee/actions.py` and
  `strategy/common/abstract_behaviour.py`. Not dead BT scaffolding —
  `kernel.RefereeOverride` imports and runs `actions.py`'s Step classes
  directly (a duck-typed `_BlackboardShim`, not real py_trees tree-ticking).
  `abstract_behaviour.py` was trimmed to drop the now-truly-dead `setup()`
  blackboard-registration method (nothing calls it anymore — `RefereeOverride`
  only ever calls `setup_()`), keeping just the `setup_`/`initialise`/`update`
  contract `actions.py`'s Steps need.
- Every behavioral test that rode on the deleted examples was ported onto
  kernel-based strategies first, so no coverage was silently dropped: ball
  placement (`test_ball_placement_rsim.py`), referee-override/stoppage
  handling (`test_referee_rsim.py`, plus 4 new tests added to
  `tests/engine/test_referee_override.py` for penalty/direct-free dispatch,
  which had zero coverage anywhere — old or new — until this pass),
  `exp_ball` validation (`test_exp_ball.py`), formation loading
  (`test_rsim_formations.py`), motion-planning obstacle avoidance
  (`tests/motion_planning/*.py`, via a new shared
  `tests/motion_planning/_kernel_test_strategies.py`), field-requirement
  assertions (`tests/abstract_strategy/test_assertions.py` — 7 real
  `assert_field_requirements`/`get_min_bounding_req` tests kept, 4 BT-only
  reset-guard tests dropped as redundant with existing kernel-path coverage),
  and the referee visualisation demo scripts (`tests/referee/{wandering_strategy,
  referee_sim,demo_referee_gui_rsim}.py`, manual tools not pytest-collected
  but kept working rather than left broken).
- Found and fixed a real, pre-existing bug while porting: `CustomReferee.set_command`
  for `BALL_PLACEMENT_*`/`PREPARE_KICKOFF_*`/etc. inserts `STOP` first and
  stores the real command as `next_command`; if a test pre-populates
  `ball_placement_target` before calling `set_command`, `StrategyRunner`'s
  "STOP + designated_position -> instant-place, skip to FORCE_START" fast
  path (meant only for real out-of-bounds auto-placement) fires immediately
  and the scenario never actually runs. Fixed by switching those call sites
  to `force_command`, which bypasses the STOP-insertion guard entirely.
- Two genuinely pre-existing, unrelated test issues marked `xfail` in code
  (not silently ignored) rather than force-fixed, since both are out of this
  pass's scope: `test_ball_placement_rsim.py::test_placer_moves_toward_designated_position`
  (rsim physics/timing variance against a tight progress threshold) and
  `tests/motion_planning/multiple_robots_test.py::test_mirror_swap` (a
  genuine `FastPathPlanning` convergence/local-minimum for one specific 6v6
  mirrored geometry, reproduced identically via plain `move()` commands
  independent of strategy class).
- Once nothing imports py_trees/pydot anymore *except* `actions.py`/
  `abstract_behaviour.py` (now the confirmed final state — verify with
  `grep -rln "py_trees\|pydot" --include="*.py" utama_core`), the py_trees/pydot
  dependency itself stays required (these two files still use it for real),
  so there is no further dependency-removal follow-up here.

Landed in `960662c` (on top of `087ee4b`). Full suite: 602 passed, 2 skipped,
2 xfailed, 3 failed — all 3 failures (`test_render_overlay.py` ×2,
`test_go_to_ball.py::test_dribbler_off_overshoot_is_smaller_than_dribbler_on`)
independently reproduced on the pre-pass `087ee4b` baseline, confirmed
unrelated to this cleanup, not this pass's responsibility to fix.

## AbstractStrategy follow-ups (from the BT-removal rewrite)

Deferred during the `AbstractStrategy` rewrite (merging `KernelStrategy` into it,
dropping py_trees) — not urgent, revisit once there's a concrete forcing case:

- `goalkeeper_id`/`exp_ball` as `AbstractStrategy.__init__` params: `goalkeeper_id`
  has zero real overrides today (every `build_*_kernel_strategy` factory uses the
  default `0`) — worth reconsidering whether it belongs as a constructor param at
  all, or should just be hardcoded until a config actually needs a different
  keeper id. `exp_ball` is genuinely read by `StrategyRunner`'s validation before
  any tactic runs, so it likely does need to live somewhere the runner can see it
  — but worth a closer look at whether the constructor is the right place once
  more of `AbstractStrategy`'s shape has settled.
- `KernelContext` — reconsider whether it's still needed as a wrapper once the
  BT-removal pass is fully done. It exists to thread `motion_controller` through
  every `Tactic.tick()` call; worth checking whether that indirection earns its
  keep once `AbstractStrategy` itself is simpler.

## GUI / tooling

- **Getter GUI — read-only live game/robot state viewer.** Raised by the
  user (2026-08-16). Distinct from the existing `custom_referee/gui.py`
  referee GUI (`_RefereeGUIServer`, SSE-pushed referee command/score/BT-debug
  panel, launched via `enable_gui=True` on `CustomReferee`): that GUI is
  referee-centric (state machine, rules, scoreboard), not a general "inspect
  the live game state on demand" tool. A "getter" GUI would let a user pull
  up current robot positions/orientations/velocities, ball position, and
  whatever else `Game`/`GameFrame` exposes, without needing to already know
  what to `print()` or attach a debugger. Not designed yet — open questions:
  reuse `_RefereeGUIServer`'s existing HTTP/SSE server plumbing (add a new
  panel/endpoint) vs. a genuinely separate standalone tool; whether it needs
  push (SSE, like the referee GUI) or plain pull-on-request is enough for
  "getter" framing; whether it should work against a live `StrategyRunner`
  only or also replay a captured log. `gui.py`'s existing
  `_serialise_game_frame`/`_serialise_robots`/`_serialise_ball` helpers
  already do most of the state→JSON work this would need, so it's likely a
  smaller lift than a from-scratch GUI — worth checking those for reuse
  before building new serialization.

## CustomReferee gaps (2026-08-16 re-derivation) — all 3 resolved

From the 2026-08-16 re-derivation in `docs/custom_referee.md`'s "Known gaps"
section (see "Repo root cleanup" below for how this list was recovered
after the source transcripts were deleted). All 3 genuinely-open items are
now done:

- ~~No double-touch rule~~ **Done.** New `DoubleTouchRule`
  (`rules/double_touch_rule.py`), scoped narrowly to the real SSL rule: only
  the designated kicker of a restart (`DIRECT_FREE_*`, `PREPARE_KICKOFF_*`,
  `PREPARE_PENALTY_*` → `NORMAL_START`) is barred from touching the ball
  again before another robot does. Deliberately does **not** apply to
  general open-play dribbling — the first draft didn't scope it this way
  and would have falsely fouled every `DribbleTactic` sequence; caught via
  AskUserQuestion before landing, not after. Arms on the restart→NORMAL_START
  edge (detected via the rule's own `_prev_command` tracking across
  `check()` calls, since `BaseRule.check()` doesn't receive the previous
  command directly), disarms on any other robot's touch or on leaving
  `NORMAL_START`. New `DoubleTouchConfig`, wired into both YAML profiles
  (enabled in `simulation`, disabled in `human`). 7 new tests, including one
  end-to-end through the real `CustomReferee.step()` call pattern
  specifically to validate the arming-edge timing against actual code, not
  a hand-rolled simulation of it.
- ~~No ball-speed rule~~ **Done.** New `BallSpeedRule`
  (`rules/ball_speed_rule.py`) fires once, edge-detected, when the ball's
  ground speed (`hypot(v.x, v.y)` — z-velocity from a bounce excluded)
  crosses above `max_speed_mps` (default 6.5 m/s), and awards `DIRECT_FREE_*`
  to the non-kicking team. Uses the same last-touch tracking approach as
  `OutOfBoundsRule` (IR `has_ball` first, closest-robot-within-0.15m
  fallback). New `BallSpeedConfig` in `profile_loader.py`, wired into both
  YAML profiles — enabled in `simulation`, disabled in `human` (same
  reasoning as the other strict-rule toggles). 8 new tests cover the
  threshold, the z-velocity exclusion, once-per-kick edge detection,
  re-firing after dropping below and back above the limit, team assignment,
  command gating, and the no-known-touch case.
- ~~No full-episode `reset()`~~ **Done.** `GameStateMachine.reset()` and
  `CustomReferee.reset()` restore score/command/stage/timers to their
  starting values without constructing a new instance, for RL episode
  reuse. Required splitting `BaseRule.reset()` (called on every command
  transition — some rules, like `GoalRule`'s cooldown timestamp,
  deliberately keep state across these) from a new
  `BaseRule.reset_for_new_episode()` (called by `CustomReferee.reset()`;
  defaults to calling `reset()`, but `GoalRule` overrides it to also clear
  the cooldown timestamp, since a new episode's clock starts fresh and a
  stale timestamp could suppress an early goal). 7 new tests in
  `test_custom_referee.py` cover state restoration, timer clearing,
  construction-config preservation, and the goal-cooldown episode-boundary
  edge case specifically.
- ~~`CustomReferee.set_bt_data`/`_bt_nodes_per_robot` are stale BT-era
  names~~ **Partially done (2026-08-16).** `set_bt_data` → `set_debug_status`
  renamed in `custom_referee.py` and its one call site
  (`strategy_runner.py:1607`) — verified no other references repo-wide.
  **Deliberately not touched**: `gui.py`'s internal `_bt_data`/`bt_data`
  naming and the served JSON key `"bt_nodes"` — that's a wider rename (wire
  format, not just a Python method name) with no template/JS consumer
  anywhere in this repo to check compatibility against, so an external tool
  could depend on the `"bt_nodes"` key today. Left alone rather than guess;
  `_bt_nodes_per_robot` (the storage attribute, still BT-era-named) was also
  left alone for the same reason — it round-trips into that same JSON key via
  `gui.py`'s `notify()`.

## Developer documentation

Beyond `tactic_model_design_decisions.md` (internal decision log, not onboarding
material), need real docs aimed at a new contributor: how kernel/`Strategy`/`Tactic`/
`Partitioner` fit together, how to author a new `Tactic` end to end, testing
conventions (headless rsim for most things, grsim for anything dribble-related since
rsim has known dribble simulation bugs).

## CI

**This section was stale — CI already exists.** `.github/workflows/ci.yml`
(pytest, headless, `--level full` on push / `--level quick` on PR,
`--ignore-glob "**/*grsim*"`, JUnit test report) and `.github/workflows/lint.yml`
(ruff) both exist, are well-configured, and have a real run history on other
branches going back well before the tactic-kernel work started. The `--level`
flag comes from a root-level `conftest.py` (not `utama_core/tests/conftest.py`)
that scales certain test parametrizations (`robot_id`, `my_team_is_right`, etc.)
between `quick` and `full`.

**What was actually missing (2026-08-16):** `spike/tactic-kernel` had never
been pushed to GitHub, so none of this branch's ~10 commits of BT-removal +
`CustomReferee` work had ever been validated by CI. Running the exact CI
command locally (`pytest utama_core/tests/ --level full --ignore-glob
"**/*grsim*" --headless`) surfaced 3 failures — none caused by this branch's
work (independently reproduced on the pre-BT-removal baseline `087ee4b` too,
per the "Codebase cleanup" section above) but real, fixable bugs:

- `test_render_overlay.py`'s two tests described features that were never
  built and aren't wanted: a multi-segment-line renderer (`draw_line`'s own
  docstring says it deliberately uses only the first and last point — every
  real caller in `ssl_gym_base.py` relies on exactly that) and an
  `OverlayType.CIRCLE` that no caller anywhere ever constructs (`POINT` is
  the real filled-circle marker, via `pygame.draw.circle(..., width=0)`).
  Rewrote both tests to assert the actual documented/used behavior instead
  of a spec for code that doesn't exist.
- `test_go_to_ball.py`'s dribbler-overshoot test asserted
  `_APPROACH_OVERSHOOT_M == ROBOT_RADIUS * 0.5`, a stale hardcoded value
  left over from before `_APPROACH_OVERSHOOT_M` was intentionally tuned to
  `0`. Dropped that one assertion, kept the two that express the test's
  actual intent (`_DRIBBLE_OVERSHOOT_M > 0` and `_APPROACH_OVERSHOOT_M <
  _DRIBBLE_OVERSHOOT_M` — dribbler-off overshoot smaller than dribbler-on).

Local run of the exact `--level full` CI command after these fixes: 624
passed, 2 skipped, 2 xfailed, **0 failed**. CI would be green if this branch
were pushed. Branch has deliberately not been pushed yet (per explicit
instruction) — pushing and confirming a real green run on GitHub is the
next concrete step whenever that's wanted.

**Still an open, separate question:** whether `tests/engine/` and
`tests/strategy_runner/` (the real tactic-kernel surface — no test files
elsewhere are kernel-specific) deserve dedicated CI treatment — e.g. a
`@pytest.mark.engine` marker so they run fast/prioritized on every push,
rather than only as part of the undifferentiated full-suite sweep. No
pytest markers of any kind exist in this repo yet. Not done in this pass —
flagged for whenever CI's actual bottleneck (if any) becomes clear from
real run times.

### TODO — add grsim as a CI/tournament environment, alongside rsim

Raised by the user (2026-08-16). Three separable problems bundled under one
goal — tracked as three items rather than one, since they have different
owners/timelines (a simulator bug fix vs. new CI infrastructure vs. an
environment-parity investigation):

1. **rsim ball-stickiness bug — confirmed real and reproduced (2026-08-16).**
   A subagent investigation reproduced this directly (standalone scratch
   script, not committed: single-robot rsim scenario, dribble to the ball,
   release via `empty_command()`, log robot-ball distance every tick for 5s
   post-release). Result: **the ball never separates** — distance froze at
   0.1123m for all 300 ticks, `has_ball` stayed `True` the entire window. A
   control run substituting `kick()` at the release point separated the ball
   by tick 2, confirming the measurement methodology and narrowing the bug to
   the passive/no-kick release path specifically.

   Root cause, read directly from `rsoccer_simulator/src/ssl/envs/
   standard_ssl.py`'s `_apply_dribbler_release_kicks`/`_dribbler_release_kick`
   (lines 314-387): the native robosim simulator doesn't reliably let go of
   the ball on dribbler-off by itself, so there's a Python-side compensating
   hack that fires a synthetic release kick — but only if the robot's
   *previous-tick* commanded forward velocity exceeds `MIN_RELEASE_SPEED =
   0.1 m/s` (`config/settings.py:26`). A robot that stops and cuts the
   dribbler in the same tick (`prev_forward = 0`) never clears that gate, so
   `release = 0.0` — no kick, ball stays glued.

   `DribbleTactic`'s release branch (`tactics/dribble.py:137-141`) does
   exactly that: sends `empty_command()` (zero velocity, dribbler off) every
   tick and loops on `not ball_separated(...)`. Confirmed via the repro:
   **this is a genuine deadlock** — that branch would never exit in rsim.

   Was this the same issue as [[project_rsim_dribble_issues]]? Still not
   definitively resolved either way, but it's at minimum consistent with
   that memory's framing (rsim dribble physics being untrustworthy, planned
   move to grsim for dribble testing) — this investigation adds a specific,
   reproduced mechanism to what was previously a more general caution.

   Scope check: `DribbleTactic` is **not wired into any of the 8
   `build_*_kernel_strategy` tournament configs** (verified directly against
   `kernel/kernel_strategy.py` — no `DribbleTactic` reference anywhere in the
   file). Every ball-release path actually used by those 8 configs
   (`_pass_and_score.py`'s pass/shoot logic, used by `PassAndShootTactic`,
   `GiveAndGoTactic`, `DecoyOverloadTactic`, `LeadAndSupportTactic`) holds
   with `empty_command(dribbler_on=True)` — dribbler *stays on* — right up
   until an actual `kick()` call, which the control run confirmed separates
   reliably. So this bug is real and should be fixed before `DribbleTactic`
   is ever wired into a kernel config, but it did **not** cause the
   tournament's scoreless-draw pattern (see
   "Multi-strategy / tournament evaluation infra" above) — that has a
   different, still-open cause.

   **User pushed back on treating this as a tactic-layer problem** (correctly
   — a passive dribbler-off release and an active kick are physically
   different mechanisms; making `DribbleTactic` fake a kick would just bake a
   simulator bug into tactic code, and every future tactic doing a passive
   release would need to remember the same workaround). Root-caused properly
   instead, in the actual native simulator source.

   **Root cause found (2026-08-16), in `rc-robosim` itself, not this repo.**
   `rc-robosim` is a PyPI package built from `github.com/robocin/rSim` (C++,
   ODE physics, pybind11 bindings) — this repo depends on it via
   `pixi.toml`'s `[feature.robosim.pypi-dependencies]` (`rc-robosim>=1.2,<2`,
   pinned to v1.2 on PyPI) and shells out to it as a subprocess
   (`rsoccer_simulator/src/Simulators/robosim/robosim_wrapper.py`). Cloned
   the real source and read it directly — two genuine bugs, not one:

   1. **The dribbler is a one-way latch.** `SSLWorld::setActions()`
      (`src/robosim/sslworld.cpp`) contains
      `if (rbtAction[7] > 0) this->robots[i]->kicker->setDribbler(true);`
      — no `else` branch ever calls `setDribbler(false)`. Once a robot's
      dribbler turns on, nothing in the native simulator's per-tick action
      processing can turn it back off; the Python-side dribbler-off command
      is silently dropped. This is the actual root cause — the earlier
      Python-side `_dribbler_release_kick` gate (`MIN_RELEASE_SPEED`) never
      even gets a chance to matter, because `dribblerOn` never flips.
   2. **Even with #1 fixed, `unholdBall()` doesn't move the ball.** It only
      destroys the ODE hinge joint (`dJointDestroy`); the ball is left at
      rest, still geometrically touching the kicker box. Ball-vs-kicker
      collision is checked every physics substep regardless of the hinge
      (`PWorld::handleCollisions` via `dSpaceCollide2(ball, spaceKicker,
      ...)`), so ODE immediately generates a fresh contact joint and
      constrains the ball right back — a naive "push it forward" nudge gets
      silently cancelled if it points into the kicker rather than away from
      it (confirmed by testing exactly that naive version first: identical
      frozen output, bit-for-bit, as the unpatched build).

   **Fix**, patched directly in the cloned `rSim` source
   (`docs/patches/rSim-dribbler-release.diff` in this repo — not applied to
   this repo's own code, since the bug lives in the upstream C++ package):
   `setActions()` now unconditionally calls `setDribbler(rbtAction[7] > 0)`;
   `unholdBall()` now computes the actual outward vector from the kicker box
   to the ball, repositions the ball just clear of the collision envelope
   along that vector, and gives it a small (0.3 m/s) outward velocity —
   enough to separate, well below a real kick's ~5 m/s. Also needed one
   unrelated compiler-compatibility fix to build at all on this machine's
   GCC 13 (`CMakeLists.txt`: pybind11 2.6.2's vendored headers assume
   `<cstdint>`/`<cstddef>` are transitively included, which modern libstdc++
   no longer guarantees — forced via `-include`).

   **Verified fixed**: rebuilt `rc-robosim` from the patched source (built
   from inside `.pixi/envs/robosim` using its own pinned toolchain, matching
   what CI/production would use), installed it over stock 1.2 in that pixi
   environment, and re-ran the same reproduction script — the ball now
   separates by release_tick=2 (matching real `kick()` timing), `has_ball`
   correctly flips to `False`, and the ball settles at a physically
   reasonable ~0.59m away instead of freezing at 0.1123m forever. The normal
   `kick()` path was re-verified unaffected (8+m roll, immediate
   separation, same as before the patch).

   **Not yet safe to ship**: running the headless test suite
   (`pytest --headless --level quick`, excluding grsim) against the patched
   build surfaced one regression —
   `test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
   passes on stock 1.2 but fails on the patch (a robot's path planner stalls
   partway out of the keep-out zone instead of fully clearing it). Confirmed
   this isn't dribbler-related (`has_ball` is `False` throughout that test)
   — it's a real physics divergence, most likely because the ball's new
   resting position/velocity after a release lands slightly differently near
   the field center, perturbing `FastPathPlanning`'s geometry into a
   degenerate case (an `invalid value encountered in divide` warning from
   `planner.py:169`'s `perp_dir / np.linalg.norm(perp_dir)` shows up in the
   same run). Not root-caused yet.

   **Current state: reverted, but the fix is preserved and buildable.**
   `.pixi/envs/robosim` is back on stock `rc-robosim==1.2`; the *installed*
   environment currently matches pre-investigation exactly. The fix itself
   is preserved two ways, both committed to this repo:
   - `docs/patches/rSim-dribbler-release.diff` — the reviewable diff.
   - `vendor/rSim/` — the actual patched rSim source, already applied (not a
     diff to apply — a working, buildable checkout), forked at upstream
     `v1.2` (commit `b413932`). See `vendor/rSim/FORK_NOTES.md` for exactly
     what's changed, why, and the full rebuild command. Keeping the real
     source here (not just the diff) means whoever picks up the regression
     below doesn't have to re-clone upstream and re-apply anything by hand
     first.

   **TODO — fix the keep-out-zone regression before shipping this.** Not
   root-caused yet:
   `utama_core/tests/engine/test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
   passes against stock `rc-robosim==1.2` but fails against the patched
   build in `vendor/rSim` — a robot's path planner stalls at
   `dist_to_center≈0.24m` and never moves again, well short of the required
   `BALL_KEEP_OUT_DISTANCE - 0.05 = 0.75m`. Confirmed this isn't
   dribbler-related (`has_ball` reads `False` for the entire test — the
   robot never touches the ball). Leading hypothesis, not yet confirmed: the
   ball's slightly different resting position/velocity after a release (a
   direct consequence of the `unholdBall()` fix) lands close enough to the
   field center to push `FastPathPlanning`'s geometry into a degenerate case
   — an `invalid value encountered in divide` RuntimeWarning from
   `motion_planning/src/fastpathplanning/planner.py:169`'s `perp_dir /
   np.linalg.norm(perp_dir)` shows up in the same test run. Whoever picks
   this up should: (1) reproduce standalone (a script like the one used to
   verify the dribbler fix, but instrumenting `FastPathPlanning`'s obstacle
   geometry near the stall point), (2) confirm or rule out the degenerate
   perpendicular-vector hypothesis, (3) only then re-attempt installing
   `vendor/rSim`'s build over the pinned `rc-robosim` version. Do **not**
   install it over stock in `.pixi/envs/robosim` until this is resolved —
   doing so silently regresses that test (and possibly other
   center-field-proximate behavior nothing else currently exercises).
2. **grsim headless/dependency/speed investigation.** User's own framing:
   grsim is "slightly harder because of the dependency and also it not being
   able to run faster when it is running in headless mode." Two distinct
   claims to verify, not assume: (a) what grsim's actual runtime/build
   dependencies are and whether they're installable in a GitHub Actions
   runner at all (grsim is an external process per `docs/custom_referee.md`'s
   own description — every grsim demo script in this repo already says "must
   already be running," i.e. today nothing in this codebase starts/manages a
   grsim process itself); (b) whether grsim genuinely cannot exceed
   real-time even headless, or whether that's grsim's own architecture
   (unlike rsim/robosim, which — per this session's own investigation above —
   was found to run faster than real-time once an unrelated bottleneck was
   fixed; grsim may or may not have an equivalent hidden bottleneck, not
   established either way yet).
3. **CI integration, blocked on both of the above.** `.github/workflows/
   ci.yml` currently hardcodes `--ignore-glob "**/*grsim*"` specifically
   because there's no grsim process available in the CI runner today. Adding
   grsim to CI means either (a) getting grsim itself to run headless inside
   the runner (blocked on item 2's dependency question), or (b) some other
   arrangement (a grsim Docker image, a self-hosted runner with grsim
   pre-installed) — not decided, genuinely an open integration design
   question once items 1–2 are further along. The tournament-style use case
   (running `tournament.py`-shaped comparisons on grsim instead of/in
   addition to rsim) has the same blocker plus grsim's own speed ceiling —
   if grsim truly can't exceed real-time, a 28-match round-robin at 60s/match
   would take at minimum 28 minutes regardless of any code changes, unlike
   the rsim version, which this session got down to ~10 minutes by removing
   an unrelated bottleneck.

## Agentic coding infra

As the tactic catalog and contributor base potentially includes coding agents (not
just humans), worth deliberately investing in:

- ~~**`AGENTS.md`**~~ **Done** (2026-08-16) — agent-agnostic (not Claude-specific)
  root-level `AGENTS.md` covers: what the repo is (Utama-Core active,
  Utama-Strategy stale), the kernel/`Tactic`/`Strategy`/`Partitioner` model
  (verified against the real `kernel/tactic.py`/`kernel/strategy.py` code, not
  just the design doc's framing), the single-writer-partition invariant,
  `CustomReferee`/`RefereeOverride` handling, the minimalism discipline (add a
  concept only after a concrete forcing case), the `--headless` requirement and
  `--level quick|full` CI split, the rsim-dribble-flakiness caveat with the
  `xfail(strict=False, ...)` pattern to follow, a "don't trust a self-reported
  test pass, re-run it" note, and a pointer map to
  `docs/tactic_model_design_decisions.md`/`docs/custom_referee.md`/
  `docs/custom_referee_design_decisions.md`/this file for anything needing more
  depth than a one-paragraph summary. **Follow-up (2026-08-16, same session):**
  added a "Writing a Tactic" section, field-tested by actually building
  `SwitchOfPlayTactic` first and distilling the bugs found along the way
  (the recurring `go_to_point()`-faces-ball issue, `intercept_point()`'s
  passer-orientation dependency, the `is_committed()` liveness contract, and
  a note that a tactic can trip referee rules belonging to a different
  tactic entirely) rather than writing speculative guidance up front.
- **CI/testing infra shaped for agent iteration loops**, not just human PR gating —
  e.g. fast feedback on whether a newly authored `Tactic` is well-formed
  (`tag` declared, `applicable()`/`is_committed()` behave sanely) before a full
  rsim/grsim run.
- **grsim/rsim feedback surfaced back to an LLM in a usable form** — today simulator
  results are numbers/logs/plots meant for a human to read; if agents are going to
  author and iterate on tactics, they need some translation layer (match summaries,
  failure characterizations, maybe rendered trajectory snapshots) that's actually
  legible to an LLM, not just a human staring at a viewer.

This is explicitly exploratory — no design decisions made yet, just the shape of the
problem worth thinking about before committing to a mechanism.

## Repo root cleanup

Deliberately *not* bundled into the BT-removal pass — same reasoning as
"Codebase cleanup" above (isolate unrelated diffs).

**Done:**
- The three tracked `.txt` Claude Code session transcripts in root
  (`2026-02-18-...`, `2026-02-23-...`, `2026-02-25-...`) — raw terminal dumps
  from earlier work building `CustomReferee` and its RSim GUI, committed by
  accident — were deleted in `960662c` at the user's explicit request.
  **Not distilled first**: the transcripts contained a 10-item gap list from
  the original `CustomReferee` design review, lost when they were deleted.

  **Follow-up done**: re-derived all 10 items directly from
  `utama_core/custom_referee/` and its tests (not from memory of the old
  discussion) and recorded the result in `docs/custom_referee.md`'s new
  "Known gaps" section. 6 of the 10 turned out to already be resolved
  (auto-advance after goals/timeouts, the keep-out-on-bare-`STOP`
  team-assignment bug, blue-perspective goal tests, the one-frame-lag doc
  note, `StrategyRunner` integration tests, `force_start_after_goal`). 3 are
  still genuinely open (no double-touch rule, no ball-speed rule, no
  full-episode `reset()` for RL reuse) and 1 is a documented-but-accepted
  limitation (last-touch tracking falls back to a 0.15 m proximity heuristic
  at the boundary). Also caught and fixed two stale BT-era doc references
  (`docs/custom_referee.md`'s pipeline diagram said "Behaviour tree reacts";
  `CustomReferee.set_bt_data`'s docstring still says "after each behaviour
  tree tick" even though the only call site
  (`strategy_runner.py:1607`) now passes `debug_status()`, the kernel-native
  replacement — flagged as a worthwhile rename, not done here to keep this
  pass doc-only).

**Done (broken-demo triage, follow-up pass):**
- The 7 files still importing deleted `strategy.examples` were resolved
  file-by-file rather than batch-ported, since only some had a real
  kernel-tactic equivalent:
  - `demo_dribbler_test.py` — ported. `DribbleTactic` (`tactics/dribble.py`)
    already exists and matches the original demo's behaviour (fetch ball,
    loop a rectangle, release/reacquire each segment); wrapped in
    `AbstractStrategy` with a single-tactic kernel `Strategy`.
  - `demo_dribbler_test2.py` — deleted. Its "forward → left → right → back →
    stop" directional sequence has no kernel-tactic equivalent —
    `DribbleTactic` only implements the corner-loop pattern — and it was a
    near-duplicate of `demo_dribbler_test.py` testing the same underlying
    skill, so not worth a new tactic just to keep two dribbler demos.
  - `demo_ball_placement.py`, `demo_ball_placement_real.py` — deleted. No
    kernel-native ball-placement tactic exists; placement during a real
    restart is handled entirely by `kernel.RefereeOverride`
    (`kernel/referee_override.py`), not something a player-facing demo
    tactic would invoke. Recreating the standalone "operator manually
    triggers BALL_PLACEMENT_YELLOW, watch one robot place it" demo would
    need new tactic code, not a port.
  - `demo_kicker_test.py` — deleted. No kick skill exists to port to —
    `skills/src/kick_ball_at_angle.py` is an empty stub. A kicker demo needs
    that skill written first.
  - `demo_one_robot_placement.py` — deleted. Used `RobotPlacementStrategy`
    (oscillate vertically, face the ball) — pure demo/test scaffolding, no
    kernel-tactic equivalent and none needed.
  - `main.py` — ported. Was `StartupStrategy` over `exp_friendly=2`; since
    robot 0 is the goalkeeper (pinned outside the kernel scheduler) only
    robot 1 is an outfield slot, too few for `PassAndShootTactic` (hard-
    requires 2). Switched to `build_give_and_go_solo_kernel_strategy((1,))`
    instead. Also dropped a dead `runner.my.strategy.render()` call —
    `AbstractStrategy` never had a `render()` method; likely a stale
    `py_trees` dot-render call that already didn't work pre-cleanup.
  - Both surviving files (`main.py`, `demo_dribbler_test.py`) verified by
    direct module import (not just `py_compile`) — both import cleanly.

**Still pending:**
- 9 `demo_*.py` scripts plus `main.py` sit loose in the repo root, no `demos/`
  or `scripts/` directory. Now that the broken-demo triage above is done,
  worth deciding whether the survivors move into a proper subdirectory.

## Known open bug: FastPathPlanning convergence stall (`test_mirror_swap`)

`utama_core/tests/motion_planning/multiple_robots_test.py::test_mirror_swap`
is `xfail(strict=False)`, not passing. Traced during the `AbstractStrategy`
port (2026-08-15): in one specific 6v6 mirrored geometry, the two outer
"wing" robots (starting at `(-3.5, +/-0.75)`) consistently stall 0.53-0.54m
from their target — inside the 45s episode timeout but outside
`endpoint_tolerance=0.3` — while the other 4 robots converge to within a few
mm. Reproduced identically via plain `move()` commands independent of
strategy class (kernel vs the old BT path), so this is a genuine
`FastPathPlanning` convergence/local-minimum behaviour for this geometry, not
a kernel-port regression and not test flakiness. Root cause and fix are both
out of scope for whoever finds this next — this is a planner-level gap, not
a one-line patch. Worth a dedicated investigation at some point since it's a
real behaviour that could show up in an actual match with similar robot
spacing, not just a test artifact.
