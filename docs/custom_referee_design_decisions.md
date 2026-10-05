# Custom Referee — Design Decisions

Decisions from the audit of `CustomReferee` against the SSL rulebook. Numbers are stable.

## Resolved

1. **`human` profile keeps `STOP` after goals** (2026-03-31) — every auto-advance is off in
   `human`, so an operator advances each stoppage. `simulation` auto-progresses.
2. **`PrepareKickoffTheirsStep` enforces own half** (2026-03-13) — after radial clearance, each
   robot's x is clamped to our half.
3. **Unknown last touch no longer defaults to yellow** — `infer_last_touch_team`
   (`rules/last_touch.py`) attributes colour-blind, and `OutOfBoundsRule` leaves the restart
   unresolved rather than favouring a colour.
4. **`KeepOutRule`'s violation count doesn't carry over** — `CustomReferee.step()` calls
   `rule.reset()` on every command transition.
5. **`BallPlacementTheirsStep` actively clears** (2026-03-13) — robots within 0.55m of the ball
   are pushed radially outward, as in `DirectFreeTheirsStep`. Line-segment clearance (ball to
   target) is deferred.
6. **`GoalRule` only fires in live play** — no change needed: the ball isn't in play during a
   stoppage, so a goal then doesn't count.
7. **Penalty positioning, partial** (2026-03-13) — non-kickers go to the touch line
   (y = ±3.0m) behind the mark. Fully off-field placement is deferred until the simulator
   supports it. Penalties stay disabled in the built-in profiles.

## Rulebook audit (2026-09-28, Division B)

Checked against the [SSL rulebook](https://robocup-ssl.github.io/ssl-rules/sslrules.html).
"Sim" rows are deliberate deviations kept for the simulator; the reason is given.

| Rule | Rulebook | Ours before | Action |
|---|---|---|---|
| Defender Too Close To Ball (§8.4.3) | 0.5 m during an opponent kick-off or free kick; non-stopping; resets the kick timer; 2 s grace per team; no automatic sanction during STOP | Stopped play and re-awarded the restart, also during penalties | Fixed: non-stopping, 2 s re-raise, off during penalties. Only the first foul of a free kick resets its clock: our cards never remove a robot, so a defender parked inside 0.5 m held a free kick for the rest of a match |
| Keeper during a penalty (§5.3.5) | Keeper on the goal line; others 1 m behind the ball; a positioning breach is a human call (§8.3.4), not Defender Too Close | `KeepOutRule` checked every defender, keeper included | Fixed by the row above: `KeepOutRule` no longer runs during penalties, so the keeper question is moot. Our non-kickers now walk round the ball to stand 1 m behind it |
| Attacker Touched Ball In Opponent Defense Area (§8.4.2) | Touching the ball while partly or fully inside; non-stopping | Presence (centre inside) stopped play with a free kick | Fixed: needs a touch, counts partial overlap (`ROBOT_RADIUS`), non-stopping, 2 s re-raise |
| Ball Speed (§8.4.2) | Over 6.5 m/s in 3D; non-stopping | Stopped play, free kick to the other team | Fixed: non-stopping. Gap: still measured on ground (x, y) speed, not 3D |
| Goal validity (§7) | No goal if the scorer committed a non-stopping foul in the last 2 s, or the ball went above 0.15 m | Any ball in the goal scored | Fixed: foul in the last 2 s turns the goal into a goal kick. Gap: ball height is not checked |
| No Progress In Game (§8.1) | 10 s without progress while both teams may play: stop, then force start | Missing | Fixed: `NoProgressRule`; STOP now continues to a queued FORCE_START once robots clear |
| Penalty kick (§5.3.5) | Still in play after 10 s: stopped, no goal, goal kick for the defenders | The normal start turned into open play | Fixed: `PenaltyTimeLimitRule`. Not implemented: the keeper's 90° deflection and ball-moving-backwards endings |
| Ball in play after a free kick (§5.4) | In play 10 s after the free kick command | `DIRECT_FREE_*` waited for the kicker forever | Fixed: `FORCE_START` after 10 s (clock restarted once, by the first Defender Too Close foul) |
| Ball in play after a kick-off (§5.4) | In play 10 s after the kick-off | `NORMAL_START` → `FORCE_START` after 10 s if the ball hasn't moved | Already correct |
| Free-kick position (§5.3.3) | ≥ 0.2 m from all lines and ≥ 1 m from either defense area, else the closest valid spot | 0.65 m from a defense area (planner margin) | Fixed: 1 m. Sim: the spot is pushed out along x only (a square corner), not to the Euclidean closest point, and lines aren't re-clamped for in-field positions |
| Throw-in, goal kick, corner kick (§6.1.1, §6.2.1–2) | Throw-in 0.2 m in from the touch line; goal kick 1 m / 0.2 m; corner kick 0.2 m / 0.2 m, in the corner | Ball placed where it crossed, 0.25 m infield, 0.5 m on both axes near a corner, then kept 1 m off the box | Fixed: over a goal line, a corner kick 0.5 m from both lines or a goal kick 1 m from the goal line, 0.5 m from the touch line (placing it where it crossed made a free kick 2 m in front of goal: 521 kicks, 150 goals in one round-robin). Sim: 0.5 m, not 0.2 m, from the touch line, and a throw-in stays 0.25 m in; 0.2 m from two lines at once let a drifting ball go straight back out (the corner-loop fix in `OutOfBoundsRule`) |
| Aimless kick (§6.2.3, Div B) | Ball returns to the kick point | Not implemented | Gap: needs kick-point tracking; not done here |
| Free kick stages (§5.3.3) | One free-kick command; the kick puts the ball in play | `DIRECT_FREE_*` preparation, then `NORMAL_START` once the kicker is ready | Sim: kept. The two stages give the auto-advance a readiness check with no human referee |
| Ball placement (§5.2) | The team places the ball | With a simulator controller, `StrategyRunner` teleports the ball to `designated_position` and force-starts | Sim: kept, so matches don't wait on physical placement. Real mode still places |
| Multiple Defenders (§8.4.1) | A non-keeper entirely inside its own area touching the ball: penalty | More than `max_defenders` (1) with centres inside, one touching | Sim: kept. The rule doesn't know which robot is the keeper |
| Double Touch (§8.2) | Kicker may not touch again before another robot; free kick from the ball position | Same | Already correct |
| Crashing | §8.4.2 | Any contact with closing speeds within 0.3 m/s of each other was a "both at fault" crash, including robots at rest against each other (6694 fouls in 231 matches; 30 of 97 goals then voided by the goal-validity rule) | Fixed: a crash needs > 1.5 m/s projected on the line between the robots; the 0.3 m/s band then compares the robots' own speeds, as TIGERs AutoReferee's `BotCollisionDetector` does. Its 0.1 s brake lookahead is modelled too (each robot's speed less 0.4 m/s, floored at 0, where TIGERs reverses it): without it a round-robin had 226 crashes to TIGERs' 8 over the same frames |
| Pushing, Keeper Held Ball, Excessive Dribbling, Robot Stop Speed, Too Close To Opponent Defense Area, Ball Placement Interference | §8.3–8.4 | Audited in `434ab29` | Already correct; not re-checked beyond thresholds (10 s, 1 m, 1.5 m/s, 0.2 m, 0.5 m, 2 s grace) |
| Boundary Crossing (§8.4.1) | Kicking the ball over the field boundary | Not implemented | Gap: rsim has no boundary wall to kick over |

## Open

8. **`TeamInfo` should be a frozen dataclass.** It is mutable, and `RefereeRefiner` stores
   `RefereeData` records that referenced the live objects, so a later `increment_score()`
   rewrote history and made `__eq__` drop new records. Workaround (2026-04-07):
   `GameStateMachine._generate_referee_data()` snapshots with `copy.copy`. Long-term: make it
   `@dataclass(frozen=True)` and replace mutations (`increment_score()`,
   `parse_referee_packet()`, ...) with `dataclasses.replace()`, including the network referee
   path. Deferred because it touches that path.
