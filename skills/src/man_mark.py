from entities.game import Game
from motion_planning.src.motion_controller import MotionController
from skills.src.utils.move_utils import move, face_ball

import numpy as np


def man_mark(
    game: Game, motion_controller: MotionController, robot_id: int, target_id: int
):
    robot = game.friendly_robots[robot_id]
    target = game.enemy_robots[target_id]
    ball_pos = (game.ball.x, game.ball.y)
    # Position with a perpendicular offset to the line between target and ball
    dx = target.p.x - ball_pos[0]
    dy = target.p.y - ball_pos[1]
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
        robot,
        (target_x, target_y),
        face_ball((robot.x, robot.y), ball_pos),
    )
    return cmd
