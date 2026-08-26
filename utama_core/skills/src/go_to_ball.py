"""`go_to_ball` — drive a robot to the ball and (optionally) pick it up with the dribbler.

The lowest-level "get the ball" primitive; every attacking tactic that needs
a robot to reach a loose or contested ball calls this rather than computing
an approach itself. By default the approach angle is opponent-aware (see
`utama_core.skills.src.shielding`) so a robot converging on a ball an enemy
is also converging on ends up shielding it from that enemy's side instead of
wedging to a stop short of the ball entirely — the exact mechanism behind
the `default_vs_lowblock` stalemate investigation
(`docs/investigation_default_vs_lowblock_stalemate.md`) before this fix.
Pass `shield=False` to opt a call site out and always approach directly from
the robot's own position instead.

See `utama_core.skills.src.shielding`'s module docstring for why this logic
was pulled out of this file rather than kept as private helpers here (in
short: it was reach-driven — "many tactics call `go_to_ball`, so fixing it
here fixes them all" — not a considered fit for a shared movement primitive,
and it has already needed one behavior-narrowing patch, `COMMIT_RANGE`, to
stop it oscillating against a mobile defender it wasn't designed against).
"""

import math
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.context import TickContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.shielding import shielded_approach_angle
from utama_core.skills.src.utils.move_utils import move

# Overshoot past ball center so the DWA keeps driving until the robot makes
# contact. The dribbler-on approach can push further; the dribbler-off approach
# just nudges past the ball center instead of settling short.
_APPROACH_OVERSHOOT_M = ROBOT_RADIUS * 0
_DRIBBLE_OVERSHOOT_M = ROBOT_RADIUS * (1 / 10)


def _target_past_ball(ball: Vector2D, approach_oren: float, overshoot_distance: float) -> Vector2D:
    if overshoot_distance <= 0.0:
        return ball

    dx = math.cos(approach_oren)
    dy = math.sin(approach_oren)
    return Vector2D(ball.x + dx * overshoot_distance, ball.y + dy * overshoot_distance)


def go_to_ball(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
    ctx: Optional[TickContext] = None,
    shield: bool = True,
) -> RobotCommand:
    """Drive `robot_id` to the ball.

    Args:
        dribble_when_near: if True (default), runs the dribbler for the whole
            approach (so it's already spinning at contact) and overshoots the
            ball by `_DRIBBLE_OVERSHOOT_M`; if False, overshoot is `_APPROACH_OVERSHOOT_M`
            (currently 0 — stop exactly at the ball, no dribbler).
        dribble_threshold: unused by this function currently; kept for
            call-site compatibility with callers that pass it positionally.
        ctx: optional `TickContext` — when its `match_log` is set, records
            which approach branch ("shield" vs "direct") was taken this call.
            Omit for callers outside a `Tactic.tick()` that don't have a `ctx`.
        shield: if True (default), approach from the far side of any
            contesting enemy (see `utama_core.skills.src.shielding`) instead
            of a straight line from the robot's own position. Pass False to
            always approach directly — e.g. a fast-break tactic racing for a
            clearly-winnable loose ball, where the shield detour only costs
            time against an enemy who isn't actually close enough to contest.
    """
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    if shield:
        approach_oren, shielding = shielded_approach_angle(game, robot, ball)
    else:
        approach_oren, shielding = robot.angle_to(ball), False

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
