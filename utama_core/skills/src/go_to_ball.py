import math

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move

# Overshoot past ball center so the DWA keeps driving until the robot makes
# contact. The dribbler-on approach can push further; the dribbler-off approach
# just nudges past the ball center instead of settling short.
_APPROACH_OVERSHOOT_M = ROBOT_RADIUS * 0.5
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
) -> RobotCommand:
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    approach_oren = robot.angle_to(ball)

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
