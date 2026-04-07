# Custom Referee — Open Design Decisions

These are deferred design decisions identified during the code audit against the SSL rulebook.
Each item describes the current behaviour, the relevant rule, and the options to choose from.

Items marked **✅ Resolved** have been implemented and are kept here for reference.

---

## 1. ✅ Human profile keeps operator-controlled STOP after goals — resolved

**Resolution (2026-03-31):** The built-in `human` profile now disables all auto-advance
transitions. After a goal or foul, the referee remains in the current stoppage until the
operator explicitly advances the game stage. The `simulation` profile remains the
auto-progressing built-in profile for simulator, testing, and RL workflows.

---

## 2. ✅ `PrepareKickoffTheirsStep` doesn't enforce own-half requirement — resolved

**Resolution (2026-03-13):** Implemented Option B. After radial clearance, each robot's x
coordinate is clamped to our own half (`max(0, x)` when right, `min(0, x)` when left).

---

## 3. `OutOfBoundsRule` unknown last-touch defaults to yellow

**File:** `utama_core/custom_referee/rules/out_of_bounds_rule.py`

**Current behaviour:**
When the ball goes out and no robot was detected touching it (`_last_touch_was_friendly = None`),
the rule awards `DIRECT_FREE_YELLOW`.

**Relevant rule (SSL):** The last-touching team loses possession (other team gets free kick).
If truly unknown, the standard is a coin flip or alternating possession — not a fixed team.

**Options:**
- **A (current, keep):** Default to yellow. Simple, predictable, slightly unfair.
- **B (alternate):** Track which team was awarded the last unknown-touch free kick and
  alternate. Fairer over many occurrences.
- **C (favor defending team):** Award to the team in whose half the ball went out. Rough
  approximation of "attacker kicked it out".

**Recommendation:** Option A is fine for a simplified system. Option C is easy to implement
and slightly more realistic if desired.

---

## 4. `KeepOutRule` violation count carries over between command changes

**File:** `utama_core/custom_referee/rules/keep_out_rule.py`

**Current behaviour:**
`_violation_count` accumulates across command changes (e.g., transitions from
`DIRECT_FREE_YELLOW` to `PREPARE_KICKOFF_BLUE`). If a robot was encroaching for 20 frames
under one command and the command changes, it only needs 10 more frames under the new command
to trigger a violation.

**Risk level:** Low — the transition cooldown (`_TRANSITION_COOLDOWN = 0.3 s`) means the
command change and any new violation are unlikely to overlap. Also, robot positions are
usually compliant after a command transition.

**Options:**
- **A (current, keep):** Accept the minor inconsistency. The persistence threshold (30 frames)
  is large enough to make false positives very unlikely.
- **B (reset on command change):** Track the previous command and reset `_violation_count`
  whenever `current_command` changes. Clean, low cost.

**Recommendation:** Option B is a one-liner fix with no downside.

---

## 5. ✅ `BallPlacementTheirsStep` has no active clearance — resolved

**Resolution (2026-03-13):** Implemented Option B. Robots within `_BALL_KEEP_DIST` (0.55 m)
of the ball are now pushed radially outward, matching the pattern used in `DirectFreeTheirsStep`.
Option C (line-segment clearance) remains deferred.

---

## 6. `GoalRule` only fires during NORMAL_START and FORCE_START

**File:** `utama_core/custom_referee/rules/goal_rule.py`

**Current behaviour:**
Goal detection is disabled during all stoppages (STOP, PREPARE_KICKOFF, etc.).

**Edge case:** If the ball rolls into a goal during a stoppage (e.g., a robot accidentally
nudges it during STOP clearance), no goal is detected.

**Relevant rule:** In SSL, the game is stopped during stoppages so the ball isn't "in play"
and a goal during a stoppage doesn't count. This is correct behaviour.

**Status:** No change needed. Documented here for clarity.

---

## 7. ✅ Penalty kick rules are incomplete — partially resolved

**Resolution (2026-03-13):** Implemented Option B. Non-kicker robots (both teams) are now
placed at `y = ±3.0 m` (touch-line boundary) rather than spread across the field.
The x-coordinate (`behind_line_x`) is unchanged — robots remain behind the penalty mark.
Full off-field placement (Option C) is deferred until the simulator supports it.
Penalty kicks remain disabled in all built-in profiles.

---

## 8. `TeamInfo` should be a frozen dataclass

**File:** `utama_core/entities/game/team_info.py`

**Current behaviour:**
`TeamInfo` is a mutable class. `GameStateMachine._generate_referee_data()` passes `self.blue_team` / `self.yellow_team` directly into `RefereeData`. `RefereeRefiner` stores these `RefereeData` objects in `_referee_records`. Because all stored records reference the same `TeamInfo` objects, a subsequent `increment_score()` call mutates the score retroactively across all historical records — which can cause `RefereeData.__eq__` to falsely consider a new record equal to the previous one, silently dropping it from `_referee_records`.

**Workaround (applied 2026-04-07):** `_generate_referee_data()` now calls `copy.copy(self.blue_team)` / `copy.copy(self.yellow_team)` to snapshot team state at the time of record creation. This fixes the aliasing issue for the `CustomReferee` path without touching `TeamInfo` or the network referee path.

**Long-term fix:** Convert `TeamInfo` to `@dataclass(frozen=True)`. Replace all in-place mutations (`increment_score()`, `parse_referee_packet()`, etc.) with `dataclasses.replace()` calls. This eliminates the aliasing hazard at the type level across the whole codebase, and gives `TeamInfo` a correct `__eq__` for free. The network referee path (`RefereeMessageReceiver` → `parse_referee_packet()`) will need updating too.

**Why deferred:** The refactor touches the network referee path which is out of scope for the referee integration PR.
