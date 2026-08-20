"""`man_mark` — shadow one enemy robot from a fixed perpendicular offset off the ball-target line.

Positions `robot_id` on the line between the ball and `target_id`, offset
0.5m perpendicular to that line rather than sitting directly on it — a
lateral standoff instead of a straight goal-side block, so the marker
doesn't collapse onto the exact ball-target line if the enemy shifts. Called
by `PressAndContainTactic` on every marker robot (`robot_ids[1:]`, one enemy
each via `_assign_markers`) while `robot_ids[0]` presses the ball carrier
via `block_attacker` — see `press_and_contain.py`.
"""

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import face_ball, move


def man_mark(game: Game, motion_controller: MotionController, robot_id: int, target_id: int):
    """Move `robot_id` to a perpendicular offset from the ball-`target_id` line, facing the ball."""
    robot = game.friendly_robots[robot_id]
    target = game.enemy_robots[target_id]
    ball_pos = game.ball.p.to_2d()
    # Position with a perpendicular offset to the line between target and ball
    dx = target.p.x - ball_pos.x
    dy = target.p.y - ball_pos.y
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    # Perpendicular offset
    offset_x = -dy * 0.5
    offset_y = dx * 0.5

    target_x = target.p.x + offset_x
    target_y = target.p.y + offset_y

    cmd = move(
        game,
        motion_controller,
        robot_id,
        Vector2D(target_x, target_y),
        face_ball(robot.p, ball_pos),
    )
    return cmd
