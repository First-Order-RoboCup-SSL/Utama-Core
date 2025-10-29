from typing import Tuple

import numpy as np

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import rotate_vector
from utama_core.motion_planning.src.common.motion_controller import MotionController

### TODO: IT IS CONTENTIOUS IF WE EVEN NEED THIS MOVE FUNCTION ANYMORE. POTENTIALLY JUST SEND DIRECTLY TO MOTION CONTROLLER ###


def move(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Vector2D,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """Calculate the robot command to move towards a target point with a specified orientation."""

    robot = game.friendly_robots[robot_id]

    (global_x_vel, global_y_vel), angular_vel = motion_controller.calculate(
        game=game,
        robot_id=robot_id,
        target_pos=target_coords,
        target_oren=target_oren,
    )

    forward_vel, left_vel = rotate_vector(global_x_vel, global_y_vel, robot.orientation)

    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick=0,
        chip=0,
        dribble=1 if dribbling else 0,
    )


def face_ball(current: Tuple[float, float], ball: Tuple[float, float]) -> float:
    """Calculate the angle to face the ball from the current position."""
    return np.arctan2(ball[1] - current[1], ball[0] - current[0])


def turn_on_spot(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """Turns the robot on the spot to face the target orientation.

    pivot_on_ball: If True, the robot will pivot on the ball, otherwise it will pivot on its own centre.
    """
    RADIUS_MODIFIER = 1.8

    turn = move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=game.friendly_robots[robot_id].p,
        target_oren=target_oren,
        dribbling=dribbling,
    )

    if game.friendly_robots[robot_id].has_ball:
        angular_vel = turn.angular_vel
        local_left_vel = -angular_vel * RADIUS_MODIFIER * ROBOT_RADIUS
        turn = turn._replace(local_left_vel=local_left_vel)

    return turn


def kick() -> RobotCommand:
    """Returns a command to kick the ball."""
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=1,
        chip=0,
        dribble=0,
    )


def empty_command(dribbler_on: bool = False) -> RobotCommand:
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=0,
        chip=0,
        dribble=dribbler_on,
    )
