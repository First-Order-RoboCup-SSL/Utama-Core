"""`go_to_ball` — drive a robot to the ball and (optionally) pick it up with the dribbler.

The lowest-level "get the ball" primitive; every attacking tactic that needs
a robot to reach a loose or contested ball calls this rather than computing
an approach itself. Not just "drive to `ball.p`": the approach angle is
opponent-aware (see `_nearest_contesting_enemy`) so a robot converging on a
ball an enemy is also converging on ends up shielding it from that enemy's
side instead of wedging to a stop short of the ball entirely — the exact
mechanism behind the `default_vs_lowblock` stalemate investigation
(`docs/investigation_default_vs_lowblock_stalemate.md`) before this fix.

Shielding stops once we're within `_COMMIT_RANGE` of the ball ourselves —
see that constant's comment. Without this, shielding against a genuinely
mobile enemy (one actively covering a shot lane, not just racing for a
loose ball) never converges: the shield target tracks the enemy's live
position every tick with no memory, so as the enemy moves to keep covering
the lane, the target keeps sliding and the approach oscillates instead of
closing. Root-caused as the `high_line_zone` regression — see
`docs/strategies.md`'s "Known open bugs".
"""

import math
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.context import KernelContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move

# Overshoot past ball center so the DWA keeps driving until the robot makes
# contact. The dribbler-on approach can push further; the dribbler-off approach
# just nudges past the ball center instead of settling short.
_APPROACH_OVERSHOOT_M = ROBOT_RADIUS * 0
_DRIBBLE_OVERSHOOT_M = ROBOT_RADIUS * (1 / 10)

# An enemy within this range of the ball is treated as "contesting" it —
# close enough that a straight-line approach from our own current position
# would converge on the enemy's body rather than open ball, the mechanism
# behind the default_vs_lowblock investigation's pin (two robots converging
# on the same point from opposite sides wedge at this rough distance apart,
# never reaching the ball itself). See `docs/investigation_default_vs_lowblock_stalemate.md`,
# fix candidate #1.
_CONTEST_RANGE = 0.5

# Once we're this close to the ball ourselves, commit to a direct approach
# instead of continuing to shield against the contesting enemy's *live*
# position. `approach_oren = contesting_enemy.angle_to(ball)` is recomputed
# fresh every tick with no memory of its own — against a genuinely mobile
# enemy (one actively covering a shot lane, not just racing for a loose
# ball, e.g. `DecoyOverloadTactic`'s "finish" phase against a real
# defender), the shield target keeps sliding and our path planner never
# converges: confirmed via instrumented match trace, `high_line_zone` vs
# `low_block` — the robot closed to 0.34m, then the shield target moved and
# it drifted back out to 0.62m, a 7.5s oscillation that ate the entire
# scoring window (see `docs/strategies.md`'s "Known open bugs"). This is not
# a persistent freeze (no per-robot memory exists at this stateless-skill
# level, and adding one would be new general-purpose state for a single
# call site) — it is a proximity gate recomputed fresh each tick from
# information already on hand, which has the same practical effect: near
# the ball, the enemy's exact position stops mattering because there is no
# more room left to route around it, so tracking it any further only
# introduces churn. Must stay smaller than `_CONTEST_RANGE` or shield mode
# would never have room to operate at all.
_COMMIT_RANGE = 0.2


def _target_past_ball(ball: Vector2D, approach_oren: float, overshoot_distance: float) -> Vector2D:
    if overshoot_distance <= 0.0:
        return ball

    dx = math.cos(approach_oren)
    dy = math.sin(approach_oren)
    return Vector2D(ball.x + dx * overshoot_distance, ball.y + dy * overshoot_distance)


def _nearest_contesting_enemy(game: Game, ball: Vector2D) -> Optional[Vector2D]:
    """Position of the closest enemy within `_CONTEST_RANGE` of the ball, if any."""
    nearest_pos, nearest_dist = None, None
    for enemy in game.enemy_robots.values():
        if enemy is None:
            continue
        dist = enemy.p.distance_to(ball)
        if dist <= _CONTEST_RANGE and (nearest_dist is None or dist < nearest_dist):
            nearest_pos, nearest_dist = enemy.p, dist
    return nearest_pos


def go_to_ball(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
    ctx: Optional[KernelContext] = None,
) -> RobotCommand:
    """Drive `robot_id` to the ball, approaching from the far side of any contesting enemy.

    Args:
        dribble_when_near: if True (default), runs the dribbler for the whole
            approach (so it's already spinning at contact) and overshoots the
            ball by `_DRIBBLE_OVERSHOOT_M`; if False, overshoot is `_APPROACH_OVERSHOOT_M`
            (currently 0 — stop exactly at the ball, no dribbler).
        dribble_threshold: unused by this function currently; kept for
            call-site compatibility with callers that pass it positionally.
        ctx: optional `KernelContext` — when its `match_log` is set, records
            which approach branch ("shield" vs "direct") was taken this call.
            Omit for callers outside a `Tactic.tick()` that don't have a `ctx`.
    """
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    contesting_enemy = _nearest_contesting_enemy(game, ball)
    shielding = contesting_enemy is not None and robot.distance_to(ball) > _COMMIT_RANGE
    if shielding:
        # Approach from the far side of the ball relative to the contesting
        # enemy — our body ends up between the enemy and the ball (a shield),
        # instead of a straight line from our own position that, against an
        # enemy also converging on the ball, wedges both robots a fixed
        # distance short of it and never actually reaches the ball (the
        # default_vs_lowblock pin).
        approach_oren = contesting_enemy.angle_to(ball)
    else:
        # Either no contesting enemy, or we're already close enough to
        # commit — see `_COMMIT_RANGE`.
        approach_oren = robot.angle_to(ball)

    if ctx is not None and ctx.match_log is not None:
        # No per-tick counter available at skill level (only `Strategy` tracks
        # that) — `sim_time` alone is enough to order/locate a trace event,
        # same key `render_around_event` already anchors on.
        ctx.match_log.trace_if_changed(
            tick=0,
            sim_time=getattr(game, "ts", 0.0),
            key=f"go_to_ball[{robot_id}].approach",
            value="shield" if shielding else "direct",
        )

    # Kicker/dribbler is on the back of the robot; approach with back facing ball.
    target_oren = (approach_oren + math.pi) % (2 * math.pi) - math.pi

    # Dribbler runs the whole approach so it is already spinning at contact.
    dribbling = dribble_when_near

    overshoot_distance = _DRIBBLE_OVERSHOOT_M if dribbling else _APPROACH_OVERSHOOT_M
    target = _target_past_ball(ball, approach_oren, overshoot_distance)

    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=target,
        target_oren=target_oren,
        dribbling=dribbling,
    )
