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
# "Known open bugs"). This is not a persistent freeze (no per-robot memory
# exists at this stateless-skill level, and adding one would be new
# general-purpose state for a single call site) — it is a proximity gate
# recomputed fresh each tick from information already on hand, which has the
# same practical effect: near the ball, the enemy's exact position stops
# mattering because there is no more room left to route around it, so
# tracking it any further only introduces churn. Must stay smaller than
# `CONTEST_RANGE` or shield mode would never have room to operate at all.
COMMIT_RANGE = 0.2


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


def shielded_approach_angle(game: Game, robot: Vector2D, ball: Vector2D) -> tuple[float, bool]:
    """Approach angle to the ball, shielding it from a contesting enemy if one is near.

    Returns `(approach_oren, shielding)` — `approach_oren` is the angle to
    approach the ball from (already `robot.angle_to(ball)` if not
    shielding), and `shielding` reports whether the shield branch was taken,
    for callers that want to log/trace it.
    """
    contesting_enemy = nearest_contesting_enemy(game, ball)
    shielding = contesting_enemy is not None and robot.distance_to(ball) > COMMIT_RANGE
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
