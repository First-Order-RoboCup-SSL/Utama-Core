from typing import Tuple

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.motion_planning.src.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import face_ball, move


def go_to_point(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Tuple[float, float],
    dribbling: bool = False,
) -> RobotCommand:
    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=target_coords,
        target_oren=face_ball(target_coords, game.ball.p),
        dribbling=dribbling,
    )
