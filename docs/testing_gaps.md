# Testing gaps

Found 2026-08-25 while implementing 7 new `custom_referee` rules (SSL
rulebook §8.3/8.4 audit) via 3 parallel agents. Each agent's own unit tests
passed; a real bug still reached the merged tree and was only caught by
re-running the *pre-existing* full suite afterward. This file records what
kind of gap let that happen, plus a few adjacent gaps noticed along the way,
so the next round of rule/rule-adjacent work doesn't rediscover the same
thing from zero.

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
test that calls `referee.step(game_frame=None, current_time=...)` to check
state-machine command transitions without a real physics frame). Fixed
with a `if game_frame is None: return None` guard. But checking why the
other 6 pre-existing rules never hit this: they don't guard against it
either — they're just never called with `game_frame=None` while gated on
an active command, because that specific scripted test only exercises
`BALL_PLACEMENT_*`, which none of the original 6 rules check. In other
words, this isn't "6 correct rules and 1 buggy one" — it's 7 rules that all
assume a non-`None` `game_frame` once their own command-gate passes, and
only one of them has ever been asked to prove otherwise. `pushing_rule.py`,
`crashing_rule.py`, and `robot_stop_speed_rule.py` all have this same latent
assumption for `NORMAL_START`/`FORCE_START`/`STOP` respectively — not fixed
here since nothing currently exercises that path for them, but worth
knowing it's there before assuming those three are hardened.

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
