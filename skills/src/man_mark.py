from entities.game import Game
from motion_planning.src.motion_controller import MotionController
from typing import Tuple
from entities.data.command import RobotCommand
from skills.src.utils.move_utils import move, face_ball


def man_mark(
    game: Game, motion_controller: MotionController, robot_id: int, target_id: int
):
    robot = game.get_robot_pos(is_yellow, robot_id)
    target = game.get_robot_pos(not is_yellow, target_id)
    ball_pos = (game.ball.x, game.ball.y)
    # Position with a perpendicular offset to the line between target and ball
    dx = target.x - ball_pos[0]
    dy = target.y - ball_pos[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    # Perpendicular offset
    offset_x = -dy * 0.5
    offset_y = dx * 0.5

    target_x = target.x + offset_x
    target_y = target.y + offset_y

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        robot,
        0,
        (target_x, target_y),
        face_ball((robot.x, robot.y), ball_pos),
    )
    return cmd
