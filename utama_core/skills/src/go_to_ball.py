from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move


def go_to_ball(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
) -> RobotCommand:
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    target_oren = robot.angle_to(ball)

    if dribble_when_near:
        distance = robot.distance_to(ball)
        dribbling = distance < dribble_threshold

    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=ball,  # (target_x, target_y),
        target_oren=target_oren,
        dribbling=dribbling,
    )
