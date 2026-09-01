"""Opponent-aware ball-approach angle — shield the ball from a contesting enemy.

Extracted out of `go_to_ball.py` (2026-08-26) rather than kept as private
helpers there. `go_to_ball` is the shared "drive to the ball" primitive
every attacking tactic calls (`PassAndShootTactic`, `GiveAndGoTactic`,
`LeadAndSupportTactic`, `DecoyOverloadTactic`, `SwitchOfPlayTactic`, ...),
but this logic was added to it for reach, not fit: the original commit
(`12bb056`) picked `go_to_ball` specifically *because* editing one shared
function patches every caller at once, not because shielding is itself a
low-level movement concern. That reach-first placement already needed one
narrowing patch (`_COMMIT_RANGE`, added when the naive version oscillated
against a mobile defender in `high_line_zone` vs `low_block` — see
`docs/strategies.md`'s "Known open bugs") — a second data point, on top of
the first, that this behavior doesn't uniformly fit every caller of
`go_to_ball` and deserves to be an explicit, overridable choice rather than
baked into the primitive. `go_to_ball(shield=False)` is that opt-out.

The original trigger (`docs/investigation_default_vs_lowblock_stalemate.md`)
was two robots racing *symmetrically* to the same point (a kickoff with no
ceremony, both arriving simultaneously) — a fairly narrow precondition. How
often that precondition still occurs, now that a real kickoff phase exists,
is an open question worth revisiting; this module makes the behavior a
first-class, independently testable/tunable unit so that question can
actually be investigated (e.g. instrument `shielded_approach_angle`'s
`shielding` return directly) without needing to first excavate it back out
of `go_to_ball`.
"""

from __future__ import annotations

from typing import Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game

# An enemy within this range of the ball is treated as "contesting" it —
# close enough that a straight-line approach from our own current position
# would converge on the enemy's body rather than open ball, the mechanism
# behind the default_vs_lowblock investigation's pin (two robots converging
# on the same point from opposite sides wedge at this rough distance apart,
# never reaching the ball itself). See `docs/investigation_default_vs_lowblock_stalemate.md`,
# fix candidate #1.
CONTEST_RANGE = 0.5

# Once we're this close to the ball ourselves, commit to a direct approach
# instead of continuing to shield against the contesting enemy's *live*
# position. The shield angle is recomputed fresh every tick with no memory
# of its own — against a genuinely mobile enemy (one actively covering a
# shot lane, not just racing for a loose ball, e.g. `DecoyOverloadTactic`'s
# "finish" phase against a real defender), the shield target keeps sliding
# and our path planner never converges: confirmed via instrumented match
# trace, `high_line_zone` vs `low_block` — the robot closed to 0.34m, then
# the shield target moved and it drifted back out to 0.62m, a 7.5s
# oscillation that ate the entire scoring window (see `docs/strategies.md`'s
# "Known open bugs"). Must stay smaller than `CONTEST_RANGE` or shield mode
# would never have room to operate at all.
#
# A bare one-sided threshold here (originally: "no per-robot memory exists
# at this stateless-skill level, and adding one would be new general-purpose
# state for a single call site") was found to actively cause a second,
# separate bug: `docs/investigation_ball_contact_orientation_divergence.md`
# traced a robot's facing error growing monotonically for 300+ ms right at
# the point its chassis distance to the ball first dips under `COMMIT_RANGE`.
# The commit-direct target (`robot.angle_to(ball)`) differs from the shield
# target by 100+ degrees in the confirmed case, which the orientation PID's
# own jump-detection (`AbstractPID._target_jumped`) correctly resets for —
# but `dist` doesn't monotonically stay under `COMMIT_RANGE` once committed:
# the robot's own approach dynamics (braking/rotation near the contact
# radius) wobble it back and forth across the boundary, re-triggering the
# shield<->direct flip (and therefore another 100+ degree target jump and
# another PID reset) every few ticks — faster than the ~300ms a single reset
# needs to actually converge. Confirmed via instrumented trace
# (`switch_of_play_vs_default.pkl`-equivalent rerun, enemy robot 1,
# t=39.6-39.9s): `shielding` flipped True->False->True twice within 0.3s
# while `dist` oscillated 0.199 -> 0.175 -> 0.200, each flip re-triggering a
# ~3 radian `target_oren` jump before the previous reset's recovery
# completed. A bare threshold cannot fix this — hysteresis is inherently
# stateful (the decision must depend on which side it last committed to) —
# so `_COMMITTED_ROBOTS` below is deliberately reintroducing the per-robot
# memory this module previously avoided, now that avoiding it has a
# confirmed cost. See `_RELEASE_RANGE` for the other half of the band.
COMMIT_RANGE = 0.2

# Once committed to a direct approach (`COMMIT_RANGE`), don't re-engage
# shielding until the robot has backed off past this larger radius. Without
# this gap, `dist` wobbling by even a few centimetres right at `COMMIT_RANGE`
# (which happens routinely — see `COMMIT_RANGE`'s docstring) flips the
# decision every few ticks. The gap only needs to comfortably exceed that
# wobble band (observed ~2.5cm in the confirmed trace); this is a generous
# multiple of that with room to spare, still well inside `CONTEST_RANGE` so
# shielding still has room to operate against a genuinely distant approach.
_RELEASE_RANGE = 0.35

# Per-robot "have we committed to a direct approach and not yet released"
# state — see `COMMIT_RANGE`'s docstring for why this module needs it now.
# Keyed by `robot_id` only (not team), matching every other caller in this
# module/its call sites (`go_to_ball`, tactics) which only ever pass
# friendly robots through here. Never grows unboundedly: at most one entry
# per robot ID actually in play.
_COMMITTED_ROBOTS: dict[int, bool] = {}


def reset_shield_state(robot_id: int) -> None:
    """Forget any committed-direct-approach state for `robot_id`.

    Call this wherever a robot's ball-approach is being restarted from
    scratch for an unrelated reason (new possession, role reassignment) so a
    stale commitment from a previous, unrelated approach doesn't suppress
    shielding on this new one. Safe to call even if no state is held (no-op).
    Primarily for tests; most call sites don't need this since a fresh
    approach with the enemy now far away simply won't re-trigger the
    `CONTEST_RANGE` check in the first place.
    """
    _COMMITTED_ROBOTS.pop(robot_id, None)


def nearest_contesting_enemy(game: Game, ball: Vector2D) -> Optional[Vector2D]:
    """Position of the closest enemy within `CONTEST_RANGE` of the ball, if any."""
    nearest_pos, nearest_dist = None, None
    for enemy in game.enemy_robots.values():
        if enemy is None:
            continue
        dist = enemy.p.distance_to(ball)
        if dist <= CONTEST_RANGE and (nearest_dist is None or dist < nearest_dist):
            nearest_pos, nearest_dist = enemy.p, dist
    return nearest_pos


def shielded_approach_angle(game: Game, robot: Vector2D, ball: Vector2D, robot_id: int) -> tuple[float, bool]:
    """Approach angle to the ball, shielding it from a contesting enemy if one is near.

    Returns `(approach_oren, shielding)` — `approach_oren` is the angle to
    approach the ball from (already `robot.angle_to(ball)` if not
    shielding), and `shielding` reports whether the shield branch was taken,
    for callers that want to log/trace it.

    `robot_id` keys the commit/release hysteresis (see `COMMIT_RANGE`'s and
    `_RELEASE_RANGE`'s docstrings) — required, not optional, since a wrong or
    reused ID would silently share hysteresis state between two different
    robots' approaches.
    """
    contesting_enemy = nearest_contesting_enemy(game, ball)
    dist = robot.distance_to(ball)

    if contesting_enemy is None:
        # Only clear the commitment once *we* are genuinely clear of the
        # ball (past `_RELEASE_RANGE`), not merely because no enemy happens
        # to be contesting it this exact tick (found live, 2026-09-02
        # tournament re-run, `counter_flow_vs_tiki_taka.pkl` t=14.7-15.4s):
        # an enemy contesting a midfield loose ball routinely steps in and
        # out of `CONTEST_RANGE` from moment to moment, not just once —
        # unconditionally popping the commitment on every exit meant the
        # *next* re-entry always restarted hysteresis from
        # `already_committed=False`, silently defeating `_RELEASE_RANGE`'s
        # whole purpose (re-running the tight `dist > COMMIT_RANGE` check
        # instead of the wide release check) and reproducing the same
        # approach/retreat oscillation `_RELEASE_RANGE` was added to
        # prevent, just gated by the enemy's in/out timing instead of our
        # own distance wobble. Gating the clear on `dist` instead (rather
        # than never clearing at all) avoids a *different* bug: nothing
        # calls `reset_shield_state` today (checked directly — no call site
        # exists outside its own tests), so an unconditional "never clear"
        # would let a stale commitment from one approach silently suppress
        # shielding on a later, unrelated one once this robot is done with
        # the current ball entirely.
        if dist > _RELEASE_RANGE:
            _COMMITTED_ROBOTS.pop(robot_id, None)
        return robot.angle_to(ball), False

    # Hysteresis: once committed to a direct approach, stay committed until
    # comfortably clear of the ball again (`_RELEASE_RANGE`), rather than
    # flipping back the moment `dist` ticks back over `COMMIT_RANGE` by a
    # centimetre. See `COMMIT_RANGE`'s docstring for the confirmed failure
    # this replaces (bare-threshold chatter repeatedly re-triggering large
    # `target_oren` jumps faster than the orientation PID could recover from
    # the previous one).
    already_committed = _COMMITTED_ROBOTS.get(robot_id, False)
    if already_committed:
        shielding = dist > _RELEASE_RANGE
    else:
        shielding = dist > COMMIT_RANGE
    _COMMITTED_ROBOTS[robot_id] = not shielding

    if shielding:
        # Approach from the far side of the ball relative to the contesting
        # enemy — our body ends up between the enemy and the ball (a shield),
        # instead of a straight line from our own position that, against an
        # enemy also converging on the ball, wedges both robots a fixed
        # distance short of it and never actually reaches the ball (the
        # default_vs_lowblock pin).
        return contesting_enemy.angle_to(ball), True

    # Either no contesting enemy, or we're already close enough to commit —
    # see `COMMIT_RANGE`.
    return robot.angle_to(ball), False
