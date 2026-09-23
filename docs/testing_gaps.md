# Testing gaps

Found 2026-08-25 while three parallel agents added 7 `custom_referee` rules (SSL §8.3/8.4
audit): every agent's own unit tests passed, yet a real bug reached the merged tree and only
the pre-existing full suite caught it. This file records the kinds of gap that allowed that,
plus adjacent ones. **Numbering is fixed** (code and docs cite gaps by number); append new
gaps, never renumber. Full writeups: `git log -p -- docs/testing_gaps.md`.

## Open

**4. No static type checking in CI.** CI runs Ruff (a linter), which cannot flag a subclass
override whose signature drifted from its abstract base — the structural fix for gap #1's
symptom. Adopting mypy/pyright is undecided given the codebase isn't typed to that standard.

**6. New rules never systematically fuzzed against real match dynamics** — partly open.
`pushing` (closed, `a3f3795`) and `keeper_held_ball` (fired 10 times in 6 full matches) are
confirmed live. `ball_placement_interference` has never fired live: across ~200 placements the
longest non-placer dwell in the 0.5m stadium was 1.0s, under the rulebook's 2s grace. Not
treated as a bug; recheck if a match produces a ≥2s linger, or build an adversarial scenario.

## Closed (one line each)

1. **Unit-testing a rule in isolation skips the interface it's called through.**
   `BaseRule.check()` gained a 4th parameter; 7 overrides kept 3 and their direct-call unit
   tests still passed (Python doesn't check override signatures). Fix: one integration test
   per §8.4 rule through `CustomReferee.step()` (`test_referee_rules_integration.py`,
   `test_ball_contest_deadlock.py`) — `de5e69b`.
2. **Foul counter / yellow card never driven end-to-end.** `test_foul_counter_end_to_end.py`
   — `de5e69b`. Non-stopping fouls don't touch the 0.3s transition cooldown, so several can
   land at one timestamp.
3. **Two rules in one tick / non-stopping vs stopping foul untested.**
   `test_referee_scan_order.py` — `de5e69b`.
5. **`game_frame=None` in a test.** No real caller passes `None`; the test was fixed to pass a
   real `GameFrame` and the dead guard removed rather than copied to other rules — `f7e9a2a`.
7. **`RobotStopSpeedRule` fouled robots obeying `StopStep`'s clearing motion.** The rule now
   exempts a robot while it is inside `BALL_KEEP_OUT_DISTANCE` — `ccb172b`
   (`test_dribble_placement_stopspeed.py`).
8. **`HALT` had no automated resume**, so a sim run tripping it froze silently.
   `StrategyRunner` now auto-resumes `HALT` → `NORMAL_START` after 5s, sim only — `ccb172b`
   (`test_halt_auto_resumes_to_normal_start_in_sim`).
9. **`ball_placement_interference` was structurally unreachable in sim**: a sim-only
   `StrategyRunner` fast path teleported the ball and jumped straight to `FORCE_START`,
   skipping `BALL_PLACEMENT_*`. Gated on `next_command` — `f0ff450`
   (`test_real_out_of_bounds_restart_reaches_ball_placement`). The physical-carry gap is
   separate (`referee_integration.md`, Open).
10. **Replay investigations used raw numbers instead of `render_window()`** — the guidance now
    lives in `STRATEGY_DEVELOPMENT.md`'s Observability section.
11. **No automated stuck-match detector.** `utama_core/replay/stuck_detector.py`'s
    `find_stuck_windows()` (`39257ee`): per-3s window, ball frozen (position std-dev) or robot
    oscillating (FFT peak fraction); excludes non-live referee states, held/shielded ball, and
    ball resting in a defense area (`ce6abe3`). Bugs it found, each with a regression test:
    `PressAndContainTactic` loose-ball and `GiveAndGoTactic` hop timeout (`d6b3ff1`);
    `shielding.shielded_approach_angle` hysteresis reset by an enemy stepping in and out of
    range (the shielding fix, `ce6abe3`); a ~360s post-restart freeze with three layered causes
    — `clear` picker holding other robots on `block` (`0e510a0`), `DecoyOverloadTactic`'s 12s
    finish timeout (`ae6a6f3`), `ball_is_loose` ignoring a legally barred enemy near our
    defense area (`10fc89c`); and a same-team two-robot ball scrum
    (`_teammate_already_has_ball()`, `a59a8e5`). Cross-tactic convergence in general is
    roadmap item 7.
