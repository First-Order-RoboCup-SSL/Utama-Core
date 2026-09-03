# Testing gaps

Found 2026-08-25 while implementing 7 new `custom_referee` rules (SSL
rulebook §8.3/8.4 audit) via 3 parallel agents. Each agent's own unit tests
passed; a real bug still reached the merged tree and was only caught by
re-running the *pre-existing* full suite afterward. This file records what
kind of gap let that happen, plus a few adjacent gaps noticed along the way,
so the next round of rule/rule-adjacent work doesn't rediscover the same
thing from zero.

Numbering is fixed — other docs link to these gaps by number — so new gaps
are appended, not renumbered, regardless of narrative chronology.

## 7. `RobotStopSpeedRule` could foul a robot for obeying `RefereeOverride`'s own STOP-clearing motion

`StopStep`/`_clear_to_legal_positions` drives any robot caught inside
`BALL_KEEP_OUT_DISTANCE` (0.8m) at STOP-entry back to a legal point at up to
`MAX_VEL`. `RobotStopSpeedRule` started its 2s grace clock the moment STOP
was observed, then fouled any robot exceeding 1.5 m/s — so a robot that
entered STOP already deep inside the keep-out zone could get fouled for
complying with its own restart command.

**Closed 2026-08-26 — `ccb172b`.** The rule now exempts a robot from the
speed check for as long as it remains inside `BALL_KEEP_OUT_DISTANCE`,
regardless of the grace clock. Tests:
`tests/custom_referee/test_dribble_placement_stopspeed.py`.

## 8. HALT has no auto-advance anywhere, and no automated harness resumes it

`DefenseAreaStoppageRule`'s 2nd-foul escalation (and any future HALT-issuing
rule) has no auto-advance path in `GameStateMachine` — correct per the SSL
rulebook (a human referee/GC must resume HALT in a real match), but no
sim/tournament harness ever resumed one either, so an automated run tripping
HALT would freeze forever with no test failure — silent non-progress, worse
than gap #6's "rule never fires" trap.

**Closed 2026-08-26 — `ccb172b`.** `StrategyRunner._run_step` now
auto-resumes HALT to `NORMAL_START` after 5.0s, only when
`sim_controller is not None` (sim-only, never on real hardware). Test:
`tests/strategy_runner/test_referee_rsim.py::test_halt_auto_resumes_to_normal_start_in_sim`.

## 1. Unit-testing a rule in isolation doesn't test the interface it's actually called through

Every new rule's tests called `rule.check(frame, geometry, command)`
directly (3 args). Mid-session, `BaseRule.check()` grew a 4th parameter
(`designated_position`); `CustomReferee.step()` always calls with 4
positional args. Python never checks a subclass override's signature
against its abstract base, so 7 rule files still had 3-parameter overrides
when merged, and every one of their own unit tests kept passing — none of
them drove the call through `CustomReferee.step()` itself, the actual
production call path.

**Closed 2026-08-26 — `de5e69b`.** One integration-shaped test per new
§8.4 rule now exists, driven through the real `CustomReferee.step()` call
path: `Pushing` in `tests/custom_referee/test_ball_contest_deadlock.py`;
the other 6 in `tests/custom_referee/test_referee_rules_integration.py`.

## 2. No test drives the foul-counter/yellow-card mechanism end-to-end

`RuleViolation.offending_teams`/`counts_toward_foul_counter` and
`TeamInfo.increment_foul_counter()` were each unit-tested in isolation, but
nothing drove 3 real fouls through `GameStateMachine._handle_foul()` in
sequence and asserted a yellow card actually lands.

**Closed 2026-08-26 — `de5e69b`.**
`tests/custom_referee/test_foul_counter_end_to_end.py` (7 tests) drives
real `RuleViolation`s through `GameStateMachine.step()` for both teams,
confirms the 3rd/6th foul awards a 2nd card, and confirms
`counts_toward_foul_counter=False`/`offending_teams=()` charge nobody.
Bonus (non-bug) finding: non-stopping fouls never update
`_last_transition_time`, so several at the same timestamp all land, unlike
stopping fouls (suppressed by the 0.3s transition cooldown).

## 3. No test exercises two rules firing in the same tick, or a non-stopping foul's interaction with a stopping one

`CustomReferee.step()`'s scan logic (first stopping violation in priority
order wins and stops the scan; a non-stopping violation found earlier isn't
suppressed by a later stopping one but also can't pre-empt it) had no
dedicated test — added specifically to support `CrashingRule`
(`is_stopping=False`) without breaking every pre-existing rule.

**Closed 2026-08-26 — `de5e69b`.**
`tests/custom_referee/test_referee_scan_order.py` (3 tests): a lone
non-stopping violation (`CrashingRule`) is recorded without changing
`referee_command`; the first stopping rule in priority order (`GoalRule`)
wins and a call-counting wrapper proves the next rule is never even
consulted that tick; two minimal stub `BaseRule`s directly prove an earlier
non-stopping violation doesn't suppress or pre-empt a later stopping one
(the real rule set's own command-gating can't produce that ordering today).

## 4. No static type checking in CI to catch signature drift automatically

CI (`.github/workflows/lint.yml`) runs Ruff only — a linter, not a type
checker, so it doesn't flag a subclass method whose signature has drifted
from its abstract base's. This is the tooling-level version of gap #1: a
type checker (mypy/pyright) would have caught all 7 mismatched `check()`
overrides for free, before any test run was needed.

**Open.** Not decided whether adopting mypy/pyright (even permissively) is
worth the cost, given none of this codebase is currently typed to that
standard — flagged as the more structural fix underlying gap #1's symptom.

## 5. `game_frame=None` handling isn't a consistently-applied convention

`BallPlacementInterferenceRule` dereferenced `game_frame.ball` without a
`None` check, breaking a test that called `referee.step(game_frame=None,
...)` — but `CustomReferee.step()`'s own signature declares
`game_frame: GameFrame`, not `Optional[GameFrame]`; no real caller ever
passes `None`. The bug was in the test, not a missing guard.

**Closed 2026-08-26 — `f7e9a2a`.** Fixed at the source: the test now passes
a minimal real `GameFrame` instead of `None`.
`BallPlacementInterferenceRule`'s now-dead `game_frame is None` guard was
removed rather than propagated to the other rules — none of them has ever
needed one. Full suite: 799 passed, 0 failed.

## 6. New rules verified in isolation and via a small live tournament, not systematically fuzzed against thresholds

A 3-match round-robin confirmed `crashing`/`defense_area_stoppage`/
`excessive_dribbling` fire in normal 6v6 play, but `pushing`,
`keeper_held_ball`, and `ball_placement_interference` never fired — their
thresholds were only verified against small hand-constructed unit-test
scenarios, not real match dynamics.

**`pushing`: closed 2026-08-26 — `a3f3795`.**
`tests/custom_referee/test_ball_contest_deadlock.py` drives the exact
traced ball-contest-deadlock geometry through the real `CustomReferee.step()`
path and confirms both that `PushingRule` fires and that `StopStep` actually
separates the pinned robots (see `docs/roadmap.md` item 11).

**`keeper_held_ball`: closed 2026-08-26.** Fired 10 times across 4 of 6
full-length live-tournament matches (`replays/gap6_validation_20260826_124653/`)
— fires correctly and sanely in real competitive play.

**`ball_placement_interference`: open.** 0 fires across the same 6-match
validation, despite ~200 `out_of_bounds` restarts correctly routing through
`BALL_PLACEMENT_*` (gap #9 below is fixed and not the cause here). Traced:
the longest continuous dwell by a non-placing robot inside the 0.5m stadium
was 1.017s, under the 2.0s grace period every time — current strategies'
robots pass near the stadium but don't linger long enough to foul. Not
treated as a bug (the 2s/0.5m values are direct rulebook constants); still
open in the sense that no live match has ever confirmed this rule firing —
worth checking again if a future tournament produces a genuine ≥2s linger,
or via a deliberately adversarial scenario if live-play confirmation is
required.

## 9. `ball_placement_interference` was structurally unreachable in every rsim/grsim run

`GameStateMachine` always routes a stopping restart through
`BALL_PLACEMENT_*` first whenever `designated_position` is set on `STOP`.
But `StrategyRunner._run_step` had a sim-only fast path (added to speed up
sim time by teleporting the ball) that, on the very first `STOP` tick with a
`designated_position` set, teleported the ball *and* jumped straight to
`FORCE_START`, skipping `BALL_PLACEMENT_*` entirely — on every restart that
carries a `designated_position`, which per the state machine's design is
every one of them. This made `ball_placement_interference` structurally
unreachable in any automated run, explaining gap #6's "never fired" as a
real bug rather than under-sampling. (Both `test_ball_placement_rsim.py`
and `test_referee_rsim.py` had already independently worked around this
exact shortcut in their own tests, without tracing it back to the rule.)

**Closed 2026-08-26 — `f0ff450`.** Gated the fast path on
`ref_data.next_command not in _BALL_PLACEMENT_COMMANDS`. Regression test:
`tests/strategy_runner/test_referee_rsim.py::test_real_out_of_bounds_restart_reaches_ball_placement`
(Scenario 6) — drives a real, rule-detected restart and asserts
`BALL_PLACEMENT_*` is actually observed; confirmed to fail pre-fix, pass
post-fix. Does **not** address the separate, still-open physical-carry gap
(robot carrying the ball via dribbler/IR sensor, untested end-to-end in
rsim) noted in `test_referee_rsim.py`'s "Future work" section — orthogonal,
about physical robot behavior rather than the referee state machine.

## 10. Replay investigation defaulted to raw numeric dumps instead of a rendering tool that already existed

A field-rendering tool (`utama_core/replay/render_window.py`'s
`render_window()`/`render_around_event()`) already existed but a goalkeeper
investigation reached for `load_frames_in_range()`'s raw per-tick numbers
instead — harder to interpret spatially and more expensive in context than
one image, and nothing said explicitly to prefer the rendered option.

**Closed 2026-08-26.** `docs/STRATEGY_DEVELOPMENT.md`'s Observability
section now states outright to default to `render_window()`/
`render_around_event()` over `load_frames_in_range()`, reserving the latter
for follow-up exact-value checks. A standing memory
(`feedback_replay_rendering`) records the same preference across sessions.

## 11. No automated detector for a "stuck" match (dead ball, oscillating robots)

Nothing in the test suite or tournament tooling flagged "the game state
hasn't meaningfully progressed in N seconds" — stuck states (a frozen ball,
robots oscillating without resolving anything) were only caught by a human
skimming a replay. User's suggestion: an FFT-based check (ball position
variance near 0 + a robot's position trace concentrated at one oscillation
frequency) could flag this automatically.

**Prototyped 2026-08-26** — `39257ee`: `utama_core/replay/stuck_detector.py`'s
`find_stuck_windows()`. Per-3s sliding window: flags "ball frozen" via
position std-dev, and per-robot "oscillating" via FFT peak-bin fraction of
non-DC spectral energy (distinguishes real oscillation from a
settling/decaying approach). Unit-tested with synthetic replays.

**Validated against real data — found and fixed two genuine multi-hundred-
second stuck states — `d6b3ff1`.**
`zone_fluid_vs_counter_press_Lk.pkl` (566s): `PressAndContainTactic`
positioned its presser relative to a stationary tracked enemy's own
position rather than the ball itself; fixed to drive straight at a fully
loose ball via `go_to_ball()` when `game.robot_with_ball is None`.
`tiki_taka_plus_vs_counter_flow_Lk.pkl` (~335s): `GiveAndGoTactic`'s
passer/receiver handshake had no timeout, so a passer held the ball
indefinitely if the receiver never became ready; fixed with a 4s
`_MAX_HOP_TICKS` timeout that falls through to the existing shoot-or-
reposition fallback. Both have regression tests in
`utama_core/tests/engine/test_all_tactics.py`. Known detector limitations
(not fixed this pass): kickoff-standstill false positives, and a merged
window's reported span isn't independently re-verified frozen throughout.

**(a) Kickoff/restart false-positive filtering — closed 2026-09-02 — `ce6abe3`.**
A 2026-09-01 sweep (40-match tournament) found the raw detector's headline
numbers uninterpretable without manual filtering: 82% of flagged windows
were ≤10s and clustered at match/restart start. `find_stuck_windows` now
excludes three classes before the frozen/oscillating check runs: non-live
referee command (kickoff/restart pauses, `live_play_fraction` default 0.9),
a robot legitimately holding/shielding the ball (`possession_fraction`
default 0.3), and the ball resting in a defense area under the referee's
own held-ball handling (`defense_area_fraction` default 0.5).

**One real bug found via the filtered sweep, fixed — `ce6abe3`.**
`counter_flow_vs_tiki_taka.pkl` t=14.7-15.4s traced to
`shielding.shielded_approach_angle`'s commit/release hysteresis
unconditionally clearing `_COMMITTED_ROBOTS` the instant no enemy was
within `CONTEST_RANGE` — against a midfield loose ball, an enemy stepping
in and out of that range repeatedly reset the hysteresis every time,
reproducing the exact approach/retreat oscillation `_RELEASE_RANGE` was
originally added to prevent, just gated by the enemy's timing instead of
the robot's own distance wobble.

**Convergence: six independent post-fix tournament samples (193 matches
total, full 17-config catalog coverage, both with and without
`--both-sides`), zero new bugs found beyond the shielding fix** — every
remaining flagged window classifies as `held_or_contested` (legitimate
possession, or the already-documented `go_to_ball` chassis-contact-without-
capture gap). Treated as converged for the codebase state at the time.

**New real bug found 2026-09-02, full-length (600s) tournament — a
cross-tactic ball-target collision after a free-kick restart, freezing the
rest of the match.** `clear_press_plus_vs_shadow_switch_LK.pkl` froze for
~360s (t=237-599s) after a restart placed the ball just outside the enemy
defense box. Three independent, layered causes, all found and fixed by
tracing directly against frame data:

- **Picker-level**: `ClearBallTactic` (robot 1, permanently `"clear"`) and
  `GiveAndGoTactic`'s carrier-fetch branch (a different robot) both called
  `go_to_ball` independently for the same physical ball, each individually
  correct in isolation but with no cross-tactic awareness of each other —
  both robots stalled at `OBSTACLE_CLEARANCE` from the ball and neither
  ever moved again. **Fixed — `0e510a0`**: `_clear_press_plus_picker`/
  `_clear_danger_picker` now hold every other free robot on `"block"`
  while a `"clear"` robot is still pinned/busy, instead of letting them
  fall through to attack/press/overload. This did not fully close the gap.
- **Engine-level**: re-running with the picker fix still froze —
  `DecoyOverloadTactic.is_committed()` only released on `mem.goal_scored`,
  no phase timeout, so a restart displacing the ball far from the
  decoy/overloader could pin those 2 robots at the *engine* level
  (`Strategy._choose_partition` keeps any committed slot outside the
  picker's `free_robots` entirely) for the rest of the match. **Fixed —
  `ae6a6f3`**: a 12s `_FINISH_TIMEOUT_TICKS` budget (matching
  `pass_and_shoot.py`'s existing `_PHASE_TIMEOUT_TICKS` pattern) resets the
  tactic to a fresh, unassigned state past the timeout.
- **Cross-team-boundary**: a third, distinct freeze in a fresh 5-strategy
  tournament (`tiki_taka_vs_zone_fluid_Rk.pkl`, t=540-599s) — ball dead
  just outside `zone_fluid`'s own defense-area boundary, so the keeper's
  `_ball_needs_retrieval` (requires the ball strictly inside the box) never
  triggered; a `tiki_taka` attacker was correctly barred from crossing the
  same boundary by legal defense-area enforcement, but `ball_is_loose`
  treated that legally-barred attacker as still "contesting" the ball, so
  neither team's retrieval logic ever sent a robot again. **Fixed —
  `10fc89c`**: `ball_is_loose` now skips an in-range enemy as a
  non-contester when the ball is near our own defense area and that enemy
  is legally barred from closing the gap (`in_own_defense_area` gained an
  optional `margin` parameter). Once `ball_is_loose` returns `True` again,
  `ShadowAndMarkTactic`'s existing retriever-assignment path handles both
  ball placements correctly with no further changes.

All three fixes independently confirmed via regression tests
(`test_clear_danger_holds_block_while_a_clearer_is_still_pinned`,
`test_clear_press_plus_holds_block_while_a_clearer_is_still_pinned`,
`test_decoy_and_overload_tactic.py`, 5 new tests in
`test_pass_and_score_geometry.py`) and via a fresh 40-match full-length
tournament sweeping clean on both freeze-catching buckets
(`defense_box: 0`, `corner_boundary: 0`). Full suite green throughout
(877 passed after the last fix, 4 skipped, 2 xfailed).

**Same-team ball scrum — a fourth, related mechanism, found while
investigating the `trajsample` planner (see `docs/roadmap.md` item 13) —
fixed.** Two robots on the same team could simultaneously register
`has_ball=True` on a loose ball, neither yielding: `GiveAndGoTactic`'s
carrier and `DecoyOverloadTactic`'s decoy converged on the same ball with
no cross-tactic awareness, the same architecture gap as the picker-level
freeze above but manifesting between two *different* tactics under a
different control scheme. **Fixed — `a59a8e5`**: a new
`_teammate_already_has_ball()` helper in `decoy_and_overload.py`, wired in
at every point the tactic decides to fetch the ball on its own initiative.
Full suite unchanged (943 passed, 4 skipped, 5 xfailed).

Performance note: `find_stuck_windows` was later sped up with a sliding
pointer instead of a full rescan, and replay loading moved to a columnar
(.npz) format — `56f2b6f`, `9ab33d0` — unrelated to the detector's
correctness findings above.
