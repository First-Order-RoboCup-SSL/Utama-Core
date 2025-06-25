from entities.game import Game
from motion_planning.src.pid import PID, TwoDPID
from typing import Tuple
from entities.data.command import RobotCommand
from skills.src.utils.move_utils import move, face_ball


def go_to_point(
    game: Game,
    pid_oren: PID,
    pid_trans: TwoDPID,
    robot_id: int,
    target_coords: Tuple[float, float],
    dribbling: bool = False,
) -> RobotCommand:

    return move(
        game=game,
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        robot_id=robot_id,
        target_coords=target_coords,
        target_oren=face_ball(target_coords, game.ball.p),
        dribbling=dribbling,
    )
