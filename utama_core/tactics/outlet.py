"""Outlet tactic: robots held in the attacking half on the weak side, ready for a counter-attack.

New tactic, not a copy of an existing one. Its robots never chase the ball and never
commit: each tick they move to a holding point just past halfway on the enemy side,
on the flank away from the ball, facing the ball. A turnover then has a receiver
ahead of the defensive line instead of only the attack slot's own robots.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    clamp_outside_enemy_defense_area,
    enemy_goal_line,
)
from utama_core.skills.src.utils.move_utils import move

_HOLD_X_FRACTION = 0.25  # of the enemy goal's x: just past the halfway line on the enemy side
_BALL_Y_FRACTION = -0.6  # the holding line sits on the side away from the ball
_EDGE_MARGIN = 0.5  # metres kept clear of the touchlines
_SPREAD = 1.5  # metres between robots if the slot ever holds more than one


def holding_points(game: Game, count: int) -> list[Vector2D]:
    """`count` holding points on the weak side of the attacking half, spread along y.

    The x comes from the enemy goal's own x, so the same code serves both attack
    directions. Points that would land in the enemy defense area are clamped out.
    """
    goal_x, _, _ = enemy_goal_line(game)
    ball = game.ball.p.to_2d()
    limit = game.field.half_width - _EDGE_MARGIN
    base_y = max(-limit, min(limit, _BALL_Y_FRACTION * ball.y))
    points = []
    for i in range(count):
        y = max(-limit, min(limit, base_y + (i - (count - 1) / 2.0) * _SPREAD))
        points.append(clamp_outside_enemy_defense_area(game, Vector2D(_HOLD_X_FRACTION * goal_x, y)))
    return points


@dataclass
class OutletMem:
    """Stateless: the holding points are recomputed from the ball every tick."""


class OutletTactic(BaseTactic[OutletMem]):
    """Holds its robots on the weak side of the attacking half, facing the ball.

    Never committed, so the scheduler can always move its robots elsewhere. Uses
    `move()` with an explicit orientation because `go_to_point()` always faces the ball.
    """

    tag = TacticTag.ATTACK
    keeps_mem_on_robot_change = True

    def initial_mem(self) -> OutletMem:
        return OutletMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: OutletMem
    ) -> tuple[dict[RobotId, RobotCommand], OutletMem]:
        ball = game.ball.p.to_2d()
        commands: dict[RobotId, RobotCommand] = {}
        for robot_id, point in zip(robot_ids, holding_points(game, len(robot_ids))):
            commands[robot_id] = move(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=robot_id,
                target_coords=point,
                target_oren=point.angle_to(ball),
            )
        return commands, mem
