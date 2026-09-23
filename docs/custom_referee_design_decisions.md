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

## Open

8. **`TeamInfo` should be a frozen dataclass.** It is mutable, and `RefereeRefiner` stores
   `RefereeData` records that referenced the live objects, so a later `increment_score()`
   rewrote history and made `__eq__` drop new records. Workaround (2026-04-07):
   `GameStateMachine._generate_referee_data()` snapshots with `copy.copy`. Long-term: make it
   `@dataclass(frozen=True)` and replace mutations (`increment_score()`,
   `parse_referee_packet()`, ...) with `dataclasses.replace()`, including the network referee
   path. Deferred because it touches that path.
