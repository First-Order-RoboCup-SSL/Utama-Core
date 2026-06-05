import math

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move

# Overshoot past ball center so the DWA keeps driving until the dribbler makes
# firm contact. Without this the planner's early-return at 1.5*ROBOT_RADIUS stops
# the robot a few cm short — fine in simulation but not enough for the IR sensor
# on real hardware.
_DRIBBLE_OVERSHOOT_M = ROBOT_RADIUS


def go_to_ball(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
) -> RobotCommand:
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    # Kicker/dribbler is on the back of the robot; approach with back facing ball.
    target_oren = (robot.angle_to(ball) + math.pi) % (2 * math.pi) - math.pi

    # Dribbler runs the whole approach so it is already spinning at contact.
    dribbling = dribble_when_near

    if dribbling:
        # Move target past ball center so the robot drives through to dribbler contact.
        dx = math.cos(target_oren + math.pi)
        dy = math.sin(target_oren + math.pi)
        target = Vector2D(ball.x + dx * _DRIBBLE_OVERSHOOT_M, ball.y + dy * _DRIBBLE_OVERSHOOT_M)
    else:
        target = ball

    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=target,
        target_oren=target_oren,
        dribbling=dribbling,
    )
