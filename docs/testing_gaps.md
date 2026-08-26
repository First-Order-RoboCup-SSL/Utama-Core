# Testing gaps

Found 2026-08-25 while implementing 7 new `custom_referee` rules (SSL
rulebook §8.3/8.4 audit) via 3 parallel agents. Each agent's own unit tests
passed; a real bug still reached the merged tree and was only caught by
re-running the *pre-existing* full suite afterward. This file records what
kind of gap let that happen, plus a few adjacent gaps noticed along the way,
so the next round of rule/rule-adjacent work doesn't rediscover the same
thing from zero.

**Update 2026-08-26**: gaps #1, #2, #3, and the Pushing-specific part of #6
are now closed — see the "Closed" note under each. #4 (static type checking)
and #5 (`game_frame=None` convention) remain open as tooling/design
decisions, not test-writing tasks. #6 remains open for the two rules
(`keeper_held_ball`, `ball_placement_interference`) that still haven't fired
in a live tournament, even though they're now integration-tested.

**Update 2026-08-26 (2)**: a full audit of the referee override/restart
machinery (asked for before proceeding to live-tournament validation of
`keeper_held_ball`/`ball_placement_interference`) found and fixed two
restart-safety issues — see gap #7 and gap #8 below.

## 7. `RobotStopSpeedRule` could foul a robot for obeying `RefereeOverride`'s own STOP-clearing motion

`StopStep`/`_clear_to_legal_positions` (`custom_referee/actions.py`) drives
any robot caught inside `BALL_KEEP_OUT_DISTANCE` (0.8m) at STOP-entry back
out to a legal point via the normal motion controller, at up to `MAX_VEL`
(2 m/s in rsim/grsim). `RobotStopSpeedRule` started its 2-second grace clock
the moment STOP was observed, then fouled any robot exceeding 1.5 m/s. A
robot that entered STOP already deep inside the keep-out zone (e.g.
mid-dribble at the ball) can genuinely still be moving — driven by the
referee's own override — past the 2s mark, so the referee could end up
penalizing a robot for complying with its own restart command. This would
show up in live play as a spurious `robot_stop_speed` foul with no real
non-compliance behind it.

**Fixed 2026-08-26.** `RobotStopSpeedRule.check()` (`rules/robot_stop_speed_rule.py`)
now exempts a robot from the speed check for as long as it remains inside
`BALL_KEEP_OUT_DISTANCE` of the ball, regardless of the grace clock — once a
robot reaches (or already was at) a legal distance, the ordinary
grace-period/speed check applies as before. Two new tests in
`tests/custom_referee/test_dribble_placement_stopspeed.py` cover both the
exemption (`test_exempt_while_still_inside_keep_out_zone_past_grace_period`)
and that the rule still fires normally once a robot is clear of the zone
(`test_fires_once_robot_clears_keep_out_zone_and_still_speeds`).

## 8. HALT has no auto-advance anywhere, and no automated harness resumes it

`DefenseAreaStoppageRule`'s 2nd-foul escalation (and any future rule that
issues `HALT`) has no auto-advance path in `GameStateMachine` — by design,
per the SSL rulebook, HALT requires a human referee/GameController to
resume. That's correct behavior for a real match. But grepping
`full_match_tournament.py`, `arena_tournament.py`, `tournament.py`, and
`StrategyRunner` found no code path that ever resumes a HALT — an automated
sim/tournament run that trips a HALT-issuing rule would freeze the match
forever with no test failure or crash, just silent non-progress. This is
exactly the kind of "silence looks like nothing happened" trap gap #6 warns
about for `keeper_held_ball`/`ball_placement_interference` — except worse,
since a wedged match wouldn't just fail to fire a rule, it would hang the
whole run.

**Fixed 2026-08-26.** `StrategyRunner._run_step` (`run/strategy_runner.py`)
now auto-resumes HALT to `NORMAL_START` after `_SIM_HALT_AUTO_RESUME_SECONDS`
(5.0s), but only when `self.sim_controller is not None` — i.e. only in
simulation, never on real hardware, where an actual human/GC is expected to
be present and this would be wrong to short-circuit. Covered by
`tests/strategy_runner/test_referee_rsim.py::test_halt_auto_resumes_to_normal_start_in_sim`,
which forces HALT via a real `CustomReferee`/rsim `StrategyRunner` and
confirms the match resumes to `NORMAL_START` rather than staying frozen.

## 1. Unit-testing a rule in isolation doesn't test the interface it's actually called through

Every new rule's tests called `rule.check(frame, geometry, command)`
directly — 3 positional args, matching `BaseRule.check()`'s signature at
the time each agent started. Mid-session, `BaseRule.check()` grew a 4th
parameter (`designated_position`, defaulted to `None`) to plumb the ball
placement target through to `BallPlacementInterferenceRule`. Concrete
subclasses in Python are never checked against their abstract base's
signature — nothing stops a subclass `check(self, a, b, c)` from
"implementing" a base class whose abstract method is `check(self, a, b, c,
d=None)`. `CustomReferee.step()` calls every rule the same way:

```python
result = rule.check(game_frame, self._geometry, self._state.command, self._state.ball_placement_target)
```

— 4 positional args, always. Seven rule files (6 pre-existing + 1 new)
still had 3-parameter `check()` overrides when the three agents' work was
merged. Every one of their own unit tests passed, because every one of
those tests called `check()` with only 3 args, matching what the test
author wrote against. Nothing in that test file ever drove the call
through `CustomReferee.step()` itself, so nothing ever supplied a 4th
argument and the mismatch stayed invisible until a full-suite run happened
to exercise `CustomReferee.step()` for an unrelated reason.

**The gap:** a new rule's test suite exercised the rule class directly, never
the actual call path (`CustomReferee.step()` → `rule.check(...)`) production
code uses. Passing in isolation said nothing about whether the rule was
correctly wired into the system that calls it.

**What would have caught it sooner:** at least one test per new rule that
goes through `CustomReferee.from_profile_name(...).step(game_frame, t)` end
to end, not just `SomeRule().check(...)`. Doesn't need to replace the
focused unit tests — those are still the right tool for exercising a rule's
actual logic/thresholds — but at least one integration-shaped test per rule
would have caught this specific class of bug immediately, and generalizes
to catching any future interface drift the same way.

**Closed 2026-08-26.** One integration-shaped test per new §8.4 rule now
exists, each driven through the real `CustomReferee.step()` call path:
`Pushing` in `tests/custom_referee/test_ball_contest_deadlock.py`; the
remaining 6 (`Crashing`, `KeeperHeldBall`, `ExcessiveDribbling`,
`RobotStopSpeed`, `BallPlacementInterference`, `DefenseAreaStoppage`) in
`tests/custom_referee/test_referee_rules_integration.py`. Two real (not
bugs, just non-obvious) wiring behaviors surfaced while writing these:
`KeeperHeldBallRule`'s foul auto-advances `STOP -> BALL_PLACEMENT_BLUE`
within the same tick when no robot is present to keep the "all clear" gate
pending, and `RobotStopSpeedRule`'s grace clock starts from the first
`step()` call that observes `STOP`, not from `force_command`'s timestamp —
both now documented in the new test file's comments.

## 2. No test drives the foul-counter/yellow-card mechanism end-to-end

`RuleViolation.offending_teams`/`counts_toward_foul_counter` and
`TeamInfo.increment_foul_counter()` (every 3rd foul → yellow card) were
added this session and are each unit-tested in isolation (a `RuleViolation`
carries the right `offending_teams`; `increment_foul_counter()` returns
`True` on the 3rd call). Nothing drives 3 real fouls through
`GameStateMachine._handle_foul()` in sequence and asserts a yellow card
actually lands on `TeamInfo.yellow_cards`. The wiring between "a rule
returns a violation with `offending_teams=(True,)`" and "the state machine
actually increments the right team's counter and awards a card on the 3rd"
is exactly the kind of connective logic that unit tests of the two
endpoints, individually, don't cover.

**Closed 2026-08-26** by `tests/custom_referee/test_foul_counter_end_to_end.py`
(7 tests) — drives real `RuleViolation`s through `GameStateMachine.step()`
in sequence for both teams, confirms the 3rd/6th foul awards a 2nd card
(not a one-shot special case), confirms `counts_toward_foul_counter=False`
and `offending_teams=()` both correctly charge nobody, and confirms a
non-stopping foul still applies its foul-counter side effect without
touching `command`. Bonus finding, not a bug: non-stopping fouls never
update `_last_transition_time`, so unlike stopping fouls they're never
suppressed by the 0.3s transition cooldown — several non-stopping
violations at the exact same timestamp all land.

## 3. No test exercises two rules firing in the same tick, or a non-stopping foul's interaction with a stopping one

`CustomReferee.step()`'s scan logic is genuinely subtle: the first
*stopping* violation (in priority order) wins and stops the scan; a
*non-stopping* violation found earlier doesn't get suppressed by a later
stopping one, but also can't pre-empt it — it's only applied if no stopping
violation is found at all that tick. This logic was added specifically to
support `CrashingRule` (`is_stopping=False`) without breaking every
pre-existing rule (`is_stopping=True` by default). It has no dedicated test
of its own: nothing constructs a frame where, say, a `CrashingRule`
violation and a `PushingRule` violation are both present on the same tick
and asserts which one actually gets applied and why. Given how easy this
kind of scan-order logic is to get subtly wrong (and how little visual
signal a wrong-but-plausible result gives), it's worth its own focused test
independent of any single rule's behavior.

**Closed 2026-08-26** by `tests/custom_referee/test_referee_scan_order.py`
(3 tests): confirms a lone non-stopping violation (real `CrashingRule`) is
recorded as `last_violation` without changing `referee_command`; confirms
the first stopping rule in priority order (`GoalRule`) wins and a
call-counting wrapper proves the next rule in order (`OutOfBoundsRule`)
is never even consulted that tick, not just that its result is unused; and
—since the real rule set's command-gating currently can't produce a
non-stopping violation earlier in list order than a same-tick stopping one
(documented in the file's module docstring: every rule sharing
`CrashingRule`'s `NORMAL_START`/`FORCE_START` gate sits before it, every
stopping rule after it only fires during stoppage commands Crashing never
checks)—exercises that specific ordering directly via two minimal stub
`BaseRule`s, proving the earlier non-stopping violation doesn't suppress or
pre-empt the later stopping one.

## 4. No static type checking in CI to catch signature drift automatically

CI (`.github/workflows/lint.yml`) runs Ruff only — a linter, not a type
checker. Ruff does not flag a subclass method whose signature has drifted
from its abstract base's (that's a type-checker's job — mypy/pyright would
flag `BaseRule.check()`'s abstract signature vs. an override that doesn't
accept the same parameters, at least under strict-enough settings). This
is the tooling-level version of gap #1: even without writing a single new
test, a type checker in CI would have caught 7 of the mismatched
`check()` overrides on the same pull request that introduced the
mismatch, before any test run was needed at all. Worth a follow-up
investigation into whether adopting mypy/pyright (even permissively at
first, given none of this codebase is currently typed to that standard) is
worth the cost — not decided here, just flagged as the more structural fix
underlying gap #1's specific symptom.

## 5. `game_frame=None` handling isn't a consistently-applied convention

`BallPlacementInterferenceRule` dereferenced `game_frame.ball` without
checking `game_frame is None` first, breaking
`test_custom_referee_set_command_accepts_scripted_metadata` (a scripted
test that called `referee.step(game_frame=None, current_time=...)` to check
state-machine command transitions without a real physics frame). Originally
"fixed" with a `if game_frame is None: return None` guard on the rule — the
wrong shape of fix, per a user correction: `CustomReferee.step()`'s own
signature declares `game_frame: GameFrame`, not `Optional[GameFrame]` — no
real caller (`StrategyRunner`) ever passes `None`, so a guard defending
against it doesn't belong scattered across every rule. The actual bug was
in the test, which was calling `step()` outside its real contract.

**Closed 2026-08-26.** Fixed at the source: the test now passes a minimal
but real `GameFrame` (`ball=None`, empty robot dicts, real `ts`/team-colour
fields) instead of `None` itself. `BallPlacementInterferenceRule`'s
now-dead `game_frame is None` guard was removed — its existing `ball is
None` check already covers the "no ball in the frame" case correctly.
`pushing_rule.py`/`crashing_rule.py`/`robot_stop_speed_rule.py` never had
this guard and still don't need one: no caller, test or production, has
ever passed `game_frame=None` to `CustomReferee.step()`. Full suite: 799
passed, 0 failed.

## 6. New rules verified in isolation and via a small live tournament, not systematically fuzzed against thresholds

A 3-match round-robin (`tiki_taka_plus`/`counter_press`/`high_press`, 2026-
08-25) confirmed the new rules fire in normal 6v6 play without crashing:
`crashing` fired 6-11 times per match (by far the most active new rule —
expected, given normal contact play), `defense_area_stoppage` 1-3 times,
`excessive_dribbling` twice in one match. But `pushing`,
`keeper_held_ball`, and `ball_placement_interference` never fired in any
of the 3 matches — their thresholds/trigger conditions are only verified
against the small hand-constructed scenarios in each rule's unit tests,
never against real match dynamics. This doesn't mean anything is wrong
with them; it means they're currently the least-validated of the 7 new
rules, and a future match/tournament run that happens to produce a
sustained push, a long defense-area ball hold, or a ball-placement
restart is worth checking specifically for whether those three fire
sanely (right team, right threshold, not spuriously) rather than assuming
silence means correctness.

**Partially closed 2026-08-26.** `pushing` specifically is no longer just
"unfired in one tournament + isolated unit tests" — a targeted regression
test (`tests/custom_referee/test_ball_contest_deadlock.py`) now drives the
*exact* traced ball-contest-deadlock geometry (two robots pinned around a
ball, symmetric force, neither dribbler registering contact) through the
real `CustomReferee.step()` call path and confirms both that `PushingRule`
fires correctly and that the resulting `STOP` command actually causes
`RefereeOverride`'s `StopStep` to drive the pinned robots apart. That's a
real scenario, not a synthetic one — see `docs/roadmap.md` item 11. Still
genuinely open: `keeper_held_ball` and `ball_placement_interference` have
never fired in *any* live match/tournament run, integration-tested or not —
a future tournament producing a long defense-area ball hold or a
ball-placement restart is still worth checking for these two specifically.

**Update 2026-08-26 (3):** found the actual root cause of why
`ball_placement_interference` specifically had never fired — see gap #9
below, now fixed. Live-tournament validation of both rules follows in the
next update once run.

## 9. `ball_placement_interference` was structurally unreachable in every rsim/grsim run

`GameStateMachine`'s own design always routes a stopping restart through
`BALL_PLACEMENT_*` first whenever `designated_position` is set on `STOP`
(confirmed by reading `_handle_foul`/`_handle_goal`:
`next_command`/`ball_placement_target` are set together, and
`next_command` is always the matching `BALL_PLACEMENT_*` command whenever
`designated_position is not None`). But `StrategyRunner._run_step` had a
sim-only fast path — added to speed up sim time by teleporting the ball
instead of waiting for a robot to physically carry it there — that raced
this: on the very first `STOP` tick with a `designated_position` set, it
teleported the ball *and* `force_command()`'d straight to `FORCE_START`,
skipping `BALL_PLACEMENT_*`'s existence entirely, not just the slow
physical-carry part of it. Since `STOP` is always observed strictly before
`BALL_PLACEMENT_*` in the same restart sequence, this fast path won the
race on *every* restart that carried a `designated_position` — which per
the state machine's design is every one of them (goals included, via
`ball_placement_target`). This made `BallPlacementInterferenceRule`
structurally unreachable in any automated sim/tournament run: it is only
ever checked while `current_command` is `BALL_PLACEMENT_*`, and that
command was never actually observed for more than zero ticks. This
explains gap #6's "never fired in 3 matches" as a real, fixable bug rather
than under-sampling — no number of additional tournament matches would
ever have made it fire.

The bug was hiding in plain sight: every test in
`tests/strategy_runner/test_ball_placement_rsim.py` already worked around
it, with an identical comment repeated 4 times — "`force_command` (not
`set_command`) ... `set_command` inserts STOP first with
`ball_placement_target` already populated, which trips StrategyRunner's
'STOP + designated_position -> instant-place and skip to FORCE_START' fast
path" — and `test_referee_rsim.py`'s Scenario 2b test had its own version:
"Inject directly — bypass OOB detection which now routes through ball
placement first." Both files' authors had already noticed the shortcut
defeats real restarts and routed around it in their own tests, without
tracing it back to why `ball_placement_interference` specifically could
never fire.

**Fixed 2026-08-26.** Gated the fast path in `strategy_runner.py` on
`ref_data.next_command not in _BALL_PLACEMENT_COMMANDS` — it now only
fires for STOP-preceded restarts that genuinely never go through ball
placement (there currently are none in the `simulation` profile, but the
guard is correct either way: it defers to whatever the state machine's own
`next_command` actually says, rather than assuming). The teleport-instead-
of-carry speedup itself is unchanged — the second branch (entering
`BALL_PLACEMENT_*`) still teleports the ball to the target immediately, so
sim time is not slowed down; what's restored is a real ~2-second window
(`_AUTO_ADVANCE_DELAY`) where `BALL_PLACEMENT_*` is the actually-observed
command, which is exactly what `BallPlacementInterferenceRule` needs to
exist in order to be checked at all.

New regression test:
`tests/strategy_runner/test_referee_rsim.py::test_real_out_of_bounds_restart_reaches_ball_placement`
(Scenario 6) drives a real (rule-detected, not `force_command`-injected)
out-of-bounds restart — ball drifts out under real velocity, one robot
planted inside `GameStateMachine`'s 0.5m ball-clear distance so `STOP`
genuinely persists for multiple ticks instead of auto-advancing within the
same tick it's entered — and asserts `BALL_PLACEMENT_YELLOW`/`BLUE` is
actually observed in `game.referee` before the restart concludes. Verified
this test fails (times out, `BALL_PLACEMENT_*` never observed) on the
pre-fix code and passes on the fix, confirming it's a real regression
guard and not a tautology. Full suite: 227 passed in
`tests/strategy_runner/`/`tests/custom_referee/` alone (1 pre-existing
xfail, unrelated); full-repo run pending as of this writing.

This does **not** fix or address the separate, still-open physical-carry
gap noted in `test_referee_rsim.py`'s "Future work" section (robot
carrying the ball via dribbler/IR sensor is untested end-to-end in rsim,
and would need motion-controller/dribbler-capture work before real
hardware deployment) — that gap is about physical robot behavior during
ball placement, orthogonal to this one, which was purely about whether the
referee *state machine* ever entered the `BALL_PLACEMENT_*` state at all
in an automated run.

**Update 2026-08-26 (4): live-tournament validation results for gap #6.**
Ran 6 full-length (600s) matches across 3 competitive pairs
(`counter_flow`/`tiki_taka`, `tiki_taka_plus`/`counter_flow`,
`zone_fluid`/`counter_press`, both sides), full suite green beforehand
(803 passed / 4 skipped / 2 xfailed), with the gap #9 fix active
(`replays/gap6_validation_20260826_124653/`):

- **`keeper_held_ball`: fired 10 times total**, across 4 of the 6 matches.
  Confirms this rule fires correctly and sanely in real competitive play —
  closing gap #6 for this rule. No further action needed here.
- **`ball_placement_interference`: fired 0 times**, despite ~200
  `out_of_bounds` restarts across the run (each now correctly routing
  through `BALL_PLACEMENT_*` per the gap #9 fix) and non-placing robots
  getting as close as 8-29mm to the ball-to-target line in 4 of 6 matches
  (well inside the 0.5m stadium). Traced why directly: in
  `counter_flow_vs_tiki_taka_RK.pkl`, the longest continuous dwell by a
  non-placing robot inside the 0.5m stadium was **1.017s** — under the
  2.0s grace period every time. This is not the gap #9 unreachability bug
  recurring (that's fixed and verified separately via the Scenario 6
  regression test) — it's a second, distinct, and much narrower reason:
  current strategies' robots pass through/near the stadium zone but don't
  *linger* there long enough to foul, most likely because
  `RefereeOverride`'s keep-out-clearing motion (or the tactics' own
  retreat behaviour) moves them out within about a second. Still
  genuinely open, but now precisely characterized rather than mysterious:
  `ball_placement_interference` is reachable and correctly implemented
  (confirmed via `test_placement_interference_and_defense_fixes.py`'s
  isolated unit tests and this session's Scenario 6 end-to-end test), but
  needs either (a) more/longer tournament sampling on the chance some
  future match produces a genuine ≥2s linger, or (b) a deliberately
  adversarial scenario (a tactic instructed to hold position near the
  placement line) if a live-play confirmation is required rather than a
  synthetic one. Not treating this as a bug to fix — the 2s grace period
  and 0.5m radius are both direct rulebook values (see the rule's
  docstring), and "current tactics don't linger" is a fact about the
  tactics, not evidence of a referee defect.

**Side finding during this validation run, fixed but tracked in
`docs/roadmap.md` instead of here** (not a `custom_referee` issue): the
`counter_flow_vs_tiki_taka_RK.pkl` replay from this same run also surfaced
a goalkeeper motion-control bug — a sustained, never-converging oscillation
around a static target, not a referee-rule problem. See `docs/roadmap.md`'s
"Goalkeeper overshoot (2026-08-24)" entry's 2026-08-26 update for the full
trace/fix (`GoalkeeperTactic` now uses its own dedicated `PIDController`
instead of the team's shared `FastPathPlanningController`).

**Update 2026-08-26 (5)**: two process gaps noted while investigating the
goalkeeper bug above, logged here rather than fixed immediately.

## 10. Replay investigation defaulted to raw numeric dumps instead of a rendering tool that already existed

The goalkeeper investigation used `replay_player.py`'s
`load_frames_in_range(path, t_start, t_end)` to hand a subagent raw
per-tick position/velocity numbers for the window in question. The real
problem wasn't that this particular investigation happened to skip a tool —
it's that a field-rendering tool for exactly this already existed
(`utama_core/replay/render_window.py`'s `render_window()`/
`render_around_event()`, matplotlib PNG of robot/ball trails over a pitch
outline, faint-to-solid oldest-to-newest) and it *still* wasn't reached for
by default. Handing an LLM investigator a wall of floating-point
coordinates is both harder to interpret spatially and considerably more
expensive in context window than one image, so the raw-dump path should
never be the first move once a rendering option exists — but nothing said
so explicitly, so the model default (reach for the data-shaped tool) won
out over the better option.

**Fixed 2026-08-26.** Two things now say this explicitly, so it isn't
lost the next time context resets:
- `docs/STRATEGY_DEVELOPMENT.md`'s Observability section now states
  outright to default to `render_window()`/`render_around_event()` over
  `load_frames_in_range()` for replay investigation, reserving the latter
  for follow-up exact-value checks once the image has localized what to
  look at.
- A standing memory (`feedback_replay_rendering`) records the same
  preference so it applies across sessions, not just within this repo's
  docs.

## 11. No automated detector for a "stuck" match (dead ball, oscillating robots)

Both this session's live-tournament validation and the earlier gap-hunting
sessions have relied on a human (or an agent manually skimming a replay) to
notice when a match has gotten into a degenerate state — most visibly, the
ball sitting motionless for an extended period while one or two robots
oscillate near it without resolving anything (e.g. two robots endlessly
contesting the same point, or a tactic stuck retrying a failed approach).
This is exactly the kind of failure a `custom_referee` rule *should*
eventually catch and restart (SSL's rulebook has multiple stall-breaking
provisions), but right now nothing in the test suite or tournament tooling
flags "the game state hasn't meaningfully progressed in N seconds" as a
signal on its own — it's only caught if a human happens to be looking at
the right replay window.

The user's suggestion: since a stuck point tends to show up as (a) the
ball's position variance collapsing to ~0 over a multi-second window, and
(b) one or more robots' positions oscillating periodically instead of
converging or making progress, a frequency-domain check (e.g. an FFT over
each tracked object's position trace in a sliding window) could flag "one
object frozen + another object periodic-not-progressing" automatically,
without needing to hand-author every specific stuck scenario as its own
rule. This is an interesting, cheap-to-prototype signal (it's exactly the
kind of steady-oscillation pattern the goalkeeper bug in this same session
produced) but has not been implemented or even prototyped yet — it's an
idea, not a validated detector.

**Prototyped 2026-08-26** as an offline analysis tool, not a live rule —
`utama_core/replay/stuck_detector.py`'s `find_stuck_windows(replay_path,
...)`. Per-3s sliding window: flags "ball frozen" when its position std-dev
is below `ball_still_tol` (0.05m default), and separately, per friendly
robot, takes an FFT of its x/y trace and flags "oscillating" when the
single strongest non-DC frequency bin holds more than
`oscillation_energy_tol` (0.8 default) of that trace's non-DC spectral
energy — a real back-and-forth oscillation concentrates energy at one
repeating frequency, whereas a settling/decaying approach spreads its
(smaller) non-DC energy thinly across many bins, so a peak-fraction
threshold tells them apart where a flat "any non-DC energy" measure did
not (confirmed by an early prototype iteration that wrongly flagged a pure
exponential-decay trace as oscillating — fixed by switching from summed
non-DC energy to peak-bin fraction). Adjacent flagged windows are merged;
merged spans under `min_duration_s` (3s default) are dropped. Unit-tested
with synthetic replays (`utama_core/tests/replay/test_stuck_detector.py`):
correctly flags a frozen-ball + 1Hz-oscillating-robot window, and correctly
does *not* flag ordinary steady play or a robot settling (decaying, not
oscillating) to a stop.

**Validated against real data, with two real findings.** Ran
`find_stuck_windows` over all 6 `replays/gap6_validation_20260826_124653/`
matches (the corpus from the gap #6/#9 validation run). After filtering out
the two known false-positive shapes (kickoff standstill; merged windows
whose reported span isn't genuinely frozen throughout — verified per-window
via `load_frames_in_range` before trusting any merge), two windows survived
as genuine, multi-hundred-second stuck states, both root-caused and fixed:

- **`zone_fluid_vs_counter_press_Lk.pkl`, t=33s→599s (566s).** Ball frozen
  (std ~1e-16) with an enemy `counter_press` robot parked directly on it,
  `NORMAL_START` in effect throughout. Root cause: `PressAndContainTactic`
  (`utama_core/tactics/press_and_contain.py`)'s presser tracks the
  ball-nearest *enemy* and, when that enemy doesn't have the ball,
  positions itself via `block_attacker`'s no-possession branch — a target
  computed 70% of the way **from that enemy's own position** toward the
  ball, not straight at the ball. In this replay, `zone_fluid` (the tracked
  enemy's team) was itself locked in an all-defense posture by
  `_zone_flow_picker`'s `_friendly_closer_to_ball` gate (a second, related
  but separate finding — see below), so the tracked robot never moved, and
  the presser's computed target never converged on the actual ball. Two
  individually-reasonable behaviors (containment; "don't chase a ball the
  opponent is closer to") combined into a ball nobody ever collected for
  the rest of the match.
  **Fixed 2026-08-26**: `PressAndContainTactic.tick()` now checks
  `game.robot_with_ball is None` (ball fully loose, nobody on either team
  possesses it) and drives the presser straight at the ball via
  `go_to_ball()` in that case, instead of computing a target relative to a
  possibly-stationary tracked enemy. Regression test:
  `test_press_and_contain_goes_straight_for_a_fully_loose_ball`
  (`utama_core/tests/engine/test_all_tactics.py`) — asserts `go_to_ball` is
  called (not `block_attacker`) when the tracked enemy is stationary and
  far from a loose ball; confirmed via `git stash` to fail (module doesn't
  even expose `go_to_ball` pre-fix) and pass post-fix.
  *Related, not separately fixed*: `_friendly_closer_to_ball`
  (`utama_core/strategy/kernel_strategy.py`)'s "unknown/losing → permanent
  all-defense" posture has no path back to attacking once the ball is
  genuinely loose rather than actively held by the opponent — it was not
  the proximate cause of this particular freeze (the presser fix above
  breaks the deadlock on its own, since the loose ball now gets collected
  regardless of which team's picker logic held it in defense), but the same
  "conservative default never re-evaluates" shape could plausibly recur
  elsewhere and is worth a closer look if another stuck instance surfaces
  without a `PressAndContainTactic` presser involved.

- **`tiki_taka_plus_vs_counter_flow_Lk.pkl`, t≈264s→599s (~335s).** Ball
  frozen at an extreme field corner, one `counter_flow` robot possessing it
  (`has_ball=True`) continuously, orientation frozen too, a second robot
  (the presumed pass receiver) also frozen in place nearby. Root cause:
  `GiveAndGoTactic`'s `_pass_exec` (`utama_core/tactics/_pass_and_score.py`)
  runs a synchronized passer/receiver handshake (both must reach position +
  orientation before either acts) with no timeout — if the receiver never
  becomes ready for any reason, the passer holds the ball indefinitely, and
  `GiveAndGoTactic.is_committed()` returns `True` for as long as
  `receiver_id is not None`, so the kernel never reassigns either robot
  either. The precise real-match geometry that stalled the receiver (an
  extreme corner position) was not exactly reproduced in a live rsim
  re-run — attempts with matching start positions produced different
  (non-frozen) outcomes, since small differences in surrounding robot
  placement change `_best_receiver`'s pick — but the mechanism (no timeout
  on the handshake) is real and sufficient on its own regardless of the
  exact trigger, matching this session's goalkeeper-fix precedent of
  sidestepping a mechanism rather than fully reproducing its exact trigger.
  **Fixed 2026-08-26**: added `_MAX_HOP_TICKS` (4s) to `GiveAndGoTactic`
  (`utama_core/tactics/give_and_go.py`) — a new `hop_ticks` counter on
  `GiveAndGoMem` tracks how long the current hop has been in flight, and
  abandoning it (`receiver_id = None`) past the timeout falls through to
  the tactic's existing shoot-or-reposition fallback (already proven to
  make progress — it's the same path used when no pass lane exists at
  all), restoring `is_committed() == False` so the kernel can reassign
  again too. Regression test:
  `test_give_and_go_abandons_a_hop_that_never_completes`
  (`utama_core/tests/engine/test_all_tactics.py`) — locks `receiver_id`
  directly against a receiver frame that never changes, runs
  `_MAX_HOP_TICKS + 1` ticks, asserts the hop is abandoned and
  `is_committed()` returns to `False`; confirmed via `git stash` to fail
  (import error — `_MAX_HOP_TICKS` doesn't exist pre-fix) and pass
  post-fix.

**Known limitations of the detector itself, found during this validation
run:**
- **Kickoff standstill false-positives.** Every match's opening ~1-5s
  window (all robots stationary pre-kickoff, ball placed and not yet
  live) also gets flagged — technically true (ball frozen, robots not
  making progress) but not a "stuck match" in the sense meant here, just
  a normal pre-kickoff pause. The detector currently has no notion of
  "referee command context," so it can't distinguish a legal pause from
  a genuine stall on this signal alone.
- **Merged-window span can outrun local verification.** The merge step
  reports `ball_std` as the *max* across all merged sub-windows, but two
  adjacent flagged 3s windows are only individually verified as
  frozen/oscillating on their own 3s slice — a long merged span (as seen
  above, 566s and ~335s) happened to be genuinely frozen throughout in
  both cases (spot-checked, and independently confirmed by finding and
  fixing a real root cause for each), but the merge logic doesn't itself
  guarantee that; a replay where the ball freezes, moves briefly, then
  re-freezes nearby could merge into one deceptively long reported window
  despite not being frozen for its full reported span. Neither limitation
  was fixed this pass — both are about the detector's own precision, not
  about whether it's useful (it found two real bugs despite them).

**Status: prototyped, validated on real data (found and fixed two genuine
stuck-match root causes), still not wired into any automated check**
(tournament run, CI, or otherwise) and still not a live in-match rule —
deliberately, since a false-positive "stuck" call during live play would
itself be a referee bug of the same shape as gap #9. Full test suite after
both fixes: 817 passed, 4 skipped, 2 xfailed — zero regressions. Next
steps, if picked up: (a) address the kickoff-standstill false positive
(e.g. only run the detector once `RefereeCommand` has been
`NORMAL_START`/`FORCE_START` for some minimum duration), (b) tighten the
merge logic to verify frozen/oscillating status holds across the full
merged span, not just its constituent windows, (c) re-run a fresh
tournament with both fixes active to confirm neither stuck pattern
recurs and to give the detector a clean corpus to validate against, (d)
revisit `_friendly_closer_to_ball`'s permanent-conservative-posture shape
(noted above) if another stuck instance surfaces without a
`PressAndContainTactic` presser involved, before considering any of this a
candidate for a tournament-level automated check.
