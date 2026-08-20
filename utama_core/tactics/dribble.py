"""Dribble tactic — corner-to-corner rectangle loop with release/reacquire segments.

Ported from `utama_strategy.functional.strategies.dribble`.

KNOWN ISSUE (likely rsim-specific, not a strategy-logic bug, per the
original spike): on the second and later dribble segments, the orientation
motion controller has been observed to diverge in rsim. rsim is known to
have existing issues specifically around dribbling — see
[[project_rsim_dribble_issues]] — so this should not be trusted as a verdict
on this tactic's correctness. Test on grsim before concluding anything is
wrong with the logic itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.shared.pass_and_score_geometry import at_target, has_ball
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import empty_command, move

# Rectangle corners as fractions of field half-dimensions, in traversal order.
_RECT_X_FRAC = 0.55
_RECT_Y_FRAC = 0.55

_BALL_SEPARATION_THRESHOLD = 0.13
_DIRECT_DRIBBLE_THRESHOLD = 0.95  # metres — dribble straight to target
_LEGAL_LIMIT = 0.8  # max legal dribble — buffer from 1.0 m rule
_EXPECTED_BALL_ROLL = 0.6  # ball roll at normalised release speed (PID_BUFFER=0.5)
_PID_BUFFER = 0.5  # distance past release point for PID deceleration
_MIN_LIMIT = 0.2  # minimum dribble segment length


def dribbled_enough(distance: float, limit: float) -> bool:
    return distance >= limit


def ball_separated(game: Game, robot_id: int, min_distance: float = 0.2) -> bool:
    robot = game.friendly_robots[robot_id]
    return robot.p.distance_to(game.ball.p.to_2d()) > min_distance


def set_dribble_segment(game: Game, robot_id: int, target: Vector2D) -> tuple[float, Vector2D]:
    """Returns (dribble_limit, segment_target) for the current position/target pair."""
    robot = game.friendly_robots[robot_id]
    dist_to_target = robot.p.distance_to(target)

    if dist_to_target < _DIRECT_DRIBBLE_THRESHOLD:
        return 1.0, target

    limit = min(_LEGAL_LIMIT, dist_to_target - _EXPECTED_BALL_ROLL)
    limit = max(_MIN_LIMIT, limit)

    direction = target - robot.p
    direction_norm = direction * (1.0 / dist_to_target)
    segment_target = robot.p + direction_norm * (limit + _PID_BUFFER)
    return limit, segment_target


def update_dribble_distance(
    game: Game, robot_id: int, dribbled_distance: float, last_point_with_ball: Optional[Vector2D]
) -> tuple[float, Optional[Vector2D]]:
    """Returns (new_dribbled_distance, new_last_point_with_ball)."""
    robot = game.friendly_robots[robot_id]
    current_point = robot.p

    if dribbled_distance == -1.0:
        return 0.0, current_point

    last_point = last_point_with_ball
    if last_point is None:
        last_point = current_point
    elif dribbled_distance == 0.0 and last_point.distance_to(current_point) > 0.5:
        last_point = current_point

    step_distance = last_point.distance_to(current_point)
    return dribbled_distance + step_distance, current_point


@dataclass
class DribbleMem:
    corner_index: int = 0
    final_target: Optional[Vector2D] = None
    dribbled_distance: float = 0.0
    dribble_limit: Optional[float] = None
    segment_target: Optional[Vector2D] = None
    last_point_with_ball: Optional[Vector2D] = None
    releasing: bool = False
    reacquiring: bool = True


def _make_corners(rect_x_frac: float, rect_y_frac: float) -> list[Vector2D]:
    L = 1.5 * rect_x_frac
    W = 1.125 * rect_y_frac
    return [
        Vector2D(L, W),
        Vector2D(L, -W),
        Vector2D(-L, -W),
        Vector2D(-L, W),
    ]


class DribbleTactic(BaseTactic[DribbleMem]):
    tag = TacticTag.MIXED

    def __init__(self, rect_x_frac: float = _RECT_X_FRAC, rect_y_frac: float = _RECT_Y_FRAC):
        self.corners = _make_corners(rect_x_frac, rect_y_frac)

    def initial_mem(self) -> DribbleMem:
        return DribbleMem(final_target=self.corners[0])

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: DribbleMem
    ) -> tuple[dict[RobotId, RobotCommand], DribbleMem]:
        robot_id = robot_ids[0]

        if mem.dribble_limit is None:
            mem.dribble_limit, mem.segment_target = set_dribble_segment(game, robot_id, mem.final_target)

        if at_target(game, robot_id, mem.final_target, tolerance=0.1):
            mem.corner_index += 1
            mem.final_target = self.corners[mem.corner_index % len(self.corners)]
            mem.dribbled_distance = 0.0
            mem.dribble_limit, mem.segment_target = set_dribble_segment(game, robot_id, mem.final_target)
            mem.releasing = False
            return {robot_id: empty_command(dribbler_on=True)}, mem

        # Priority: release > acquire > dribble.
        if mem.releasing or dribbled_enough(mem.dribbled_distance, mem.dribble_limit):
            mem.releasing = True
            if not ball_separated(game, robot_id, _BALL_SEPARATION_THRESHOLD):
                return {robot_id: empty_command()}, mem
            mem.dribbled_distance = -1.0
            mem.releasing = False
            return {robot_id: empty_command()}, mem

        if not has_ball(game, robot_id):
            mem.reacquiring = True
            return {
                robot_id: go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, ctx=ctx)
            }, mem

        if mem.reacquiring:
            mem.dribble_limit, mem.segment_target = set_dribble_segment(game, robot_id, mem.final_target)
            mem.reacquiring = False

        mem.dribbled_distance, mem.last_point_with_ball = update_dribble_distance(
            game, robot_id, mem.dribbled_distance, mem.last_point_with_ball
        )
        robot = game.friendly_robots[robot_id]
        command = move(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_coords=mem.segment_target,
            target_oren=robot.p.angle_to(mem.segment_target),
            dribbling=True,
        )
        return {robot_id: command}, mem
