from entities.game import Game
from motion_planning.src.pid import PID, TwoDPID
from entities.data.command import RobotCommand
from skills.src.utils.move_utils import move
import numpy as np


def go_to_ball(
    game: Game,
    pid_oren: PID,
    pid_trans: TwoDPID,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
) -> RobotCommand:
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    target_oren = robot.angle_to(ball)

    # target_x = ball_data.x - ROBOT_RADIUS * np.cos(target_oren)
    # target_y = ball_data.y - ROBOT_RADIUS * np.sin(target_oren)

    if dribble_when_near:
        distance = robot.distance_to(ball)
        dribbling = distance < dribble_threshold

    return move(
        game=game,
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        robot_id=robot_id,
        target_coords=ball,  # (target_x, target_y),
        target_oren=target_oren,
        dribbling=dribbling,
    )
