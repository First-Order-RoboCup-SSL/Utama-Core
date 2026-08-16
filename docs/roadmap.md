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
3. **CI** can now be scoped for real — cleanup is done, the tree the CI gate
   should protect actually exists.
4. Tournament/multi-strategy infra still waits on there being enough of a
   tactic catalog for comparisons to mean anything.
5. Agentic coding infra (`AGENTS.md` at least) can start any time — cheap,
   and best written while the tactic-kernel reasoning is still fresh.

## Multi-strategy / tournament evaluation infra

Today's evaluation is strategy-vs-strategy via `StrategyRunner` (one strategy per
side). We'll eventually want something tournament-shaped: many strategies (or many
tactic-kernel configs) round-robined or bracketed against each other, with aggregate
results, not just a single head-to-head match. Much later priority — revisit once
there's enough of a tactic catalog to make comparisons meaningful.

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

## CustomReferee — remaining gaps

From the 2026-08-16 re-derivation in `docs/custom_referee.md`'s "Known gaps"
section (see "Repo root cleanup" below for how this list was recovered
after the source transcripts were deleted):

- No double-touch rule (a robot touching the ball twice in a row, before
  another robot touches it, should foul).
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

No CI exists yet. At minimum: lint/format check (black, matching local pre-commit
hooks), and a headless test run (`pixi run pytest --headless`, matching local
convention — never run simulator/integration tests without `--headless`). Needs a
decision on scope — which test directories are stable enough to gate merges on,
given known rsim dribble flakiness — and probably GitHub Actions given the org
(`First-Order-RoboCup-SSL`) already lives on GitHub.

## Agentic coding infra

As the tactic catalog and contributor base potentially includes coding agents (not
just humans), worth deliberately investing in:

- **`AGENTS.md`** (agent-agnostic, not Claude-specific) — durable context a coding
  agent needs before touching this repo: the kernel/Tactic/Partitioner model, the
  single-writer-partition invariant, `--headless` requirement, minimalism discipline
  (add concepts only after a concrete forcing case), where design rationale lives
  (`docs/tactic_model_design_decisions.md`).
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
