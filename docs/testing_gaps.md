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
