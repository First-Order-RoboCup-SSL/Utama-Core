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

## Multi-strategy / tournament evaluation infra

~~Much later priority~~ **First pass done** (2026-08-16). The catalog reached 8
`build_*_kernel_strategy` configs (7 original + `decoy_and_overload`), enough
to make round-robin comparisons meaningful — `demo_tournament.py` round-robins
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

**Scoreless-draw pattern (24/28 matches, 86%) — cause still open.** Hypothesis
that this was caused by rsim's ball-stickiness-on-release bug was investigated
and **ruled out**: that bug is real (see item 1 of the grsim TODO section
below, confirmed via direct reproduction) but only affects `DribbleTactic`'s
passive-release path, which isn't wired into any of the 8 tournament configs —
every scoring/passing path those configs actually use goes through `kick()`,
confirmed to separate the ball reliably. So the tournament's scorelessness and
the rsim stickiness bug are both real but separate problems. Not yet
investigated: shot-selection/shot-lane logic (`find_best_shot`/
`segment_blocked` in `shared/pass_and_score_geometry.py`), goalkeeper
effectiveness, or whether 60s is simply too short for multi-phase tactics
(lure/drag windows, up to `_MAX_HOPS_PER_POSSESSION=6` pass hops) to reach a
shot at all — plausible directions, none confirmed.

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
  otherwise (already the pattern inside `two_robot_attack.py` and generalized into the
  kernel's `Strategy` tick loop).
- Conflict detection: never allow the same robot to be double-assigned in one tick
  silently — assert loudly instead of last-write-wins.
- An allocator/partitioner is cleanest as a pure function `(Game, ...) -> assignment`,
  re-run every tick; "static" allocation is just the trivial case of a function that
  ignores `Game`.

## More tactics — football-inspired plays/formations

Only a handful of tactics exist today (goalkeeper, defense, two_robot_attack,
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
  `tests/kernel/test_referee_override.py` for penalty/direct-free dispatch,
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
- `CustomReferee.set_bt_data`/`_bt_nodes_per_robot` are stale BT-era names
  — the only call site (`strategy_runner.py:1607`) already passes
  `debug_status()`, the kernel-native replacement. Cosmetic, but worth a
  rename (`set_bt_data` → `set_debug_status`) across `custom_referee.py`,
  `gui.py`, and that one call site next time this area is touched.

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

**Still an open, separate question:** whether `tests/kernel/` and
`tests/strategy_runner/` (the real tactic-kernel surface — no test files
elsewhere are kernel-specific) deserve dedicated CI treatment — e.g. a
`@pytest.mark.kernel` marker so they run fast/prioritized on every push,
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
   (`_pass_and_score.py`'s pass/shoot logic, used by `TwoRobotAttackTactic`,
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
   `utama_core/tests/kernel/test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
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
   (running `demo_tournament.py`-shaped comparisons on grsim instead of/in
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
  depth than a one-paragraph summary.
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
    robot 1 is an outfield slot, too few for `TwoRobotAttackTactic` (hard-
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
