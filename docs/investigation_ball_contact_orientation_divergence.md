# Investigation — a robot pinned against the ball rotates away from contact

2026-08-31. **Status: mechanism localized, not fixed.** Needs motion-controller work, not a
tactic patch. Full original writeup:
`git log -p -- docs/investigation_ball_contact_orientation_divergence.md`.

## Finding

A robot approaching via `go_to_ball` can sit at chassis contact distance
(`ROBOT_RADIUS + BALL_RADIUS` ≈ 0.1115m) for seconds (up to 11s) while its facing error
*grows* instead of converging, so `has_ball` stays `False`. rsim's `isTouchingBall()`
(`vendor/rSim/src/robosim/sslrobot.cpp`) requires the ball inside a narrow box (~3cm forward,
~4cm lateral) at the kicker face in the robot's current heading, not just within chassis
radius. Visually the robot "has the ball" but doesn't.

Example (`switch_of_play_vs_default.pkl`, friendly robot 1, t=39.0-40.5s): facing error falls
to 4° at 0.19m, then once distance locks at 0.1136m it rises steadily 25° → 48° over 0.8s —
sustained divergence once translation is blocked, not jitter.

Incidence (runs of ≥20 ticks with distance < 0.20m and no `has_ball`): 17, 28 and 21 windows
in three replays, both teams, several tactics (`LeadAndSupportTactic`,
`PressAndContainTactic`, ...). Systemic to `go_to_ball`'s short-range approach.

`move()` passes `angular_vel` straight through from the motion controller, so the divergence
originates inside the controller (orientation PID and/or the planner's lookahead carrot) once
translation stalls.

## Fix candidates (in order)

1. **Check for a `target_oren` discontinuity first.** `shielded_approach_angle`
   (`skills/src/shielding.py`) recomputes the target heading every tick; a flip near contact
   would hit the PID with a jump. `MotionController.reset(robot_id)` and the auto-reset on
   orientation discontinuity (`91100ff`) exist for exactly that shape, and shielding later
   gained commit/release hysteresis (`8059abc`). Trace `target_oren` tick by tick through a
   window like the one above before anything else.
2. Otherwise, log the carrot point and orientation-PID setpoint under near-zero translation
   (the carrot may flip sides once the robot is closer than the lookahead distance).
3. Once within the final-approach radius, compute `target_oren` directly from robot→ball
   geometry, bypassing the carrot. Smaller, but risks masking the controller bug.

(An adjacent hazard noted at the time — `has_ball()` misreading enemy ids — is now documented
as friendly-only by design in `shared/pass_and_score_geometry.py`.)
