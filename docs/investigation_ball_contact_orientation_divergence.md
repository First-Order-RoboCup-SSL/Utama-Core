# Investigation — a robot pinned against the ball rotates *away* from contact instead of settling

Date: 2026-08-31
Status: **Root-caused (mechanism localized), not fixed.** Confirmed common and
multi-second across both teams and several tactics. Needs motion-controller
work (PID/angular-velocity coupling under blocked translation), not a
tactic-level patch — out of scope for this pass; documented so the next
person doesn't have to re-derive incidence or mechanism from scratch.

## Summary

Any robot approaching the ball via `go_to_ball` (`utama_core/skills/src/go_to_ball.py`)
can end up chassis-distance-pinned at the physical contact radius
(`ROBOT_RADIUS + BALL_RADIUS ≈ 0.1115m`) while its **facing error keeps growing**
instead of converging — for multiple seconds, sometimes 10+. The robot visually
looks like it is sitting right on the ball (and often is, by chassis distance),
but `has_ball` (the real per-tick contact sensor, backed by rsim's
`isTouchingBall()`) stays `False` the whole time, because that check requires
the ball to sit inside a narrow **forward-facing kicker box**, not just within
chassis radius — and the robot is rotating steadily away from the alignment
that box needs.

This was first spotted visually (screenshot review of `replays/verify_givego_fix/
tiki_taka_vs_counter_press.pkl` at t≈12s: an enemy `PressAndContainTactic` presser
sitting on the ball, apparently turning to receive/pass, that never actually
registers possession) and confirmed to be a general, common pattern, not a
one-off.

## Mechanism

`isTouchingBall()` (`vendor/rSim/src/robosim/sslrobot.cpp:127-144`) measures
the ball's position relative to the **kicker box**, not the chassis center —
the kicker sits ~8cm forward of the chassis, per `isNearKickerFace`'s own
comment in the same file. The check requires:

```cpp
kx += vx * getKickerThickness() * 0.5f;   // kicker face offset along facing direction
...
return (xx < getKickerThickness()*2 + ballRadius) && (yy < getKickerWidth()*0.5f) && (zz < getKickerHeight()*0.5f);
```

— a narrow box (~3.15cm forward, ~4cm lateral) flush against the kicker face,
in the robot's *current facing direction*. A robot can be at exactly the right
chassis-to-ball radial distance and still fail this check if it isn't also
correctly oriented toward the ball.

Confirmed directly against `replays/switch_of_play_fixed2/switch_of_play_vs_default.pkl`,
friendly robot 1 (`LeadAndSupportTactic`'s `attack` slot), t=39.0-40.5s:

| ts | dist (m) | facing error (deg) | has_ball |
|---|---|---|---|
| 39.00 | 0.3412 | 60.4 | False |
| 39.30 | 0.3005 | 16.1 | False |
| **39.50** | 0.1935 | **4.1** (best alignment) | False |
| 39.70 | 0.1110 | 25.0 | False |
| 39.80 | 0.1134 | 28.9 | False |
| 40.00 | **0.1136** (pinned) | 33.2 | False |
| 40.20 | 0.1136 | 38.6 | False |
| 40.50 | 0.1136 | 48.2 | False |

The robot converges to near-perfect alignment (4.1°) while still ~0.19m out,
then as chassis distance locks to the contact floor (0.1136m) at t≈39.7s,
facing error **grows monotonically and continuously** — 25.0° → 28.9° → 30.7°
→ ... → 48.2° over the next 0.8s, with no sign of correcting. This is not
jitter/noise around a setpoint; it's sustained divergence once translation is
blocked by the ball.

`move()` (`utama_core/skills/src/utils/move_utils.py:15-43`) passes
`angular_vel` straight through from `motion_controller.calculate()`
(`FastPathPlanningController`, `utama_core/motion_planning/src/controllers/
fastpathplanning.py`) with no visible coupling/clamping logic at the call
site — the divergence originates somewhere inside the PID/path-planner
"carrot" lookahead once translation stalls, not in `move()` itself. **Not
traced further** — this needs someone to step through
`FastPathPlanningController.calculate()`'s orientation PID
(`utama_core.motion_planning.src.pid`) specifically for the blocked-translation
case; that's real controller-internals work, out of scope for this pass.

**Strong prior lead, not yet checked against this specific finding:** see
[[project_motion_controller_reset]] — `MotionController.reset(robot_id)`
exists specifically to clear a robot's angular PID `pre_errors`/`integrals`
when `target_oren` jumps discontinuously between ticks, because the
derivative term otherwise amplifies the jump into exactly this shape of
multi-second non-convergent spin (root-caused in a `counter_press`
investigation, 2026-08-21). `go_to_ball`'s `shielded_approach_angle`
(`utama_core/skills/src/shielding.py`) recomputes `target_oren` fresh every
tick from live enemy/robot positions with — by its own docstring — "no
memory of its own," and neither `go_to_ball` nor `lead_and_support.py` (the
tactic in the confirmed example above) ever call `reset()`. This is a
strong candidate for the *actual* mechanism, not a new controller bug:
`target_oren` plausibly flips (shield ↔ direct, or jitters as `COMMIT_RANGE`
is crossed) right as the robot nears contact, handing the PID a
discontinuous target with no reset — exactly the documented failure shape.
**Should be checked first**, before doing fresh PID-internals archaeology:
trace `target_oren`/`shielding` tick-by-tick through one of the confirmed
windows above and see if a jump lines up with where facing error starts
growing; if so, the fix is calling `ctx.motion_controller.reset(robot_id)`
at that specific transition, not new controller work at all.

## Incidence (how common, not a one-off)

Checked via: for every robot (both teams), every contiguous run of ticks
where `distance_to_ball < 0.20m` AND `has_ball == False` for ≥20 consecutive
ticks (≈0.33s at 60Hz).

| Replay | Windows found | Longest | Windows dipping below 0.1115m (contact radius) |
|---|---|---|---|
| `tiki_taka_vs_counter_press.pkl` (`verify_givego_fix`) | 17 | 7.75s | 5 |
| `switch_of_play_vs_default.pkl` (`switch_of_play_fixed2`) | 28 (independently recomputed: 28) | 11.20s (independently recomputed: 11.2s, `enemy robot 5`) | ~8-12 |
| `split_shape_vs_default.pkl` (`gui_overlay_demo_fixed`) | 21 | — | several |

Both teams, multiple tactics (`attack`/`LeadAndSupportTactic`, `press`/
`PressAndContainTactic`, unlabeled enemy tactics) — this is systemic to
`go_to_ball`'s approach behavior at short range, not one tactic's bug. The
`switch_of_play_vs_default.pkl` window count (28, longest 11.2s) was
independently recomputed against the raw replay by a second pass and
matches.

## Adjacent bug found in passing (not yet exploited, worth flagging)

`has_ball(game, robot_id, ...)` (`utama_core/shared/pass_and_score_geometry.py:31-44`)
hardcodes `game.friendly_robots[robot_id]` at all three internal lookups —
there is no team parameter. Calling it with an enemy robot ID reads the wrong
dict (`KeyError`, or silently the wrong robot if IDs happen to overlap
0-5 across teams). No current call site appears to do this (this
investigation read `Robot.has_ball` directly off frame objects rather than
through this function specifically to avoid tripping it), but the function
itself is unsafe for enemy robot IDs and should be checked/guarded before
anyone calls `has_ball(game, enemy_id)` expecting it to work.

## Why this matters

This directly explains the "robot looks like it has the ball but doesn't"
visual confusion that prompted this investigation, and is a plausible
contributing factor in some of this session's other stuck-match findings
(e.g. `SwitchOfPlayTactic`'s `switch`-phase timeout — see the "trace-verify
a fix's mechanism" work earlier this session — may be compounded by this
orientation divergence on top of the already-fixed kick-contact-gate issue,
though that was not re-checked against this specific mechanism).

## Fix candidates (not implemented, ranked by likely effort)

1. **Check whether this is just the known missing-`reset()` pattern
   recurring** (see "Strong prior lead" above) — trace `target_oren`/
   `shielding` tick-by-tick through a confirmed window and look for a jump
   coinciding with facing-error growth starting; if found, call
   `ctx.motion_controller.reset(robot_id)` on that transition. Cheapest to
   check, and matches a previously-diagnosed, previously-fixed-elsewhere
   failure shape exactly — do this before #2.
2. **If #1 doesn't explain it: investigate `FastPathPlanningController`'s
   orientation PID specifically under blocked/near-zero translation.** Some
   interaction between the path-planner's lookahead "carrot" point (which
   may itself become unstable or flip sides once the robot is closer to the
   target than the lookahead distance) and the orientation PID reacting to
   a moving/unstable reference. Needs someone to log the carrot point and
   orientation-PID setpoint tick-by-tick through one of the reproduced
   windows above.
3. **Decouple final-approach orientation from the path-planner carrot.**
   Once within some small radius of the ball (e.g. `go_to_ball`'s own
   overshoot distance), compute `target_oren` directly from robot-to-ball
   geometry rather than whatever the planner's carrot implies — bypassing
   whatever instability exists in that handoff. Smaller, more targeted than
   #2, but risks masking rather than fixing the actual controller bug.
4. **Fix the `has_ball` enemy-robot-ID hazard** (separate from the main
   finding, cheap, no behavior change for existing callers) — add a
   `is_friendly`/team argument or an assertion, so the function fails loudly
   instead of silently misreading if someone calls it with an enemy ID.
