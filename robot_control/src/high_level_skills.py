import numpy as np
from typing import Tuple

from entities.game import Game
from entities.data.command import RobotCommand
from entities.data.vision import VisionBallData

from motion_planning.src.pid import PID
from global_utils.math_utils import distance

from robot_control.src.utils.motion_planning_utils import calculate_robot_velocities
from robot_control.src.skills import get_dribble_target_candidate, go_to_ball


class DribbleToTarget:
    """
    Dribble the ball to a target location using repeated cycles of releasing,
    repositioning, and reabsorbing the ball.

    Parameters:
        pid_oren (PID): PID controller for orientation.
        pid_trans (PID): PID controller for translation.
        this_robot_data (RobotData): Current robot state (x, y, orientation).
        robot_id (int): ID of the robot.
        target_coords (Tuple[float, float]): Target x, y coordinates.
        dribble_speed (float): Speed of the dribbler. Defaults to 3.0.
        tolerance (float): Tolerance for reaching the target. Defaults to 0.1 meters.
    Returns:
        None
    """

    def __init__(
        self,
        robot_id: int,
        target_coords: Tuple[float, float],
        dribble_speed: float = 3.0,
        tolerance: float = 0.1,
        augment: bool = False,
        cooldown_sec: float = 0.25,
    ):
        self.robot_id = robot_id
        self.target_coords = target_coords
        self.dribble_speed = dribble_speed
        self.tolerance = tolerance

        self.max_dribble_d = 0.8
        self.last_point_with_ball = None
        self.dribbled_distance = 0  # TODO: this needs to be stored as a global robot state in the future (because it is accessed by all functions involving movement)
        self.dribbling_cooldown = 0
        self.cooldown_sec = cooldown_sec
        self.paused = False
        self.augment = augment

    def enact(self, game: Game, pid_oren: PID, pid_trans: PID):

        if self.dribbling_cooldown == 0:
            target_coords = get_dribble_target_candidate(
                game, self.robot_id, safe_distance=1.0
            )
            if target_coords:
                self.update_coord(target_coords)
        else:
            self.dribbling_cooldown -= 1

        this_robot = game.friendly_robots[self.robot_id]
        current_x, current_y = this_robot.p.x, this_robot.p.y
        has_ball = (
            distance((current_x, current_y), (game.ball.p.x, game.ball.p.y)) < 0.12
        )

        target_x, target_y = self.target_coords
        self._update_dribble_distance(
            (current_x, current_y), has_ball
        )  # update distance traveled with ball

        if (
            distance((current_x, current_y), self.target_coords) <= self.tolerance
            or self.dribbling_cooldown != 0
        ):
            return RobotCommand(
                local_forward_vel=0,
                local_left_vel=0,
                angular_vel=0,
                kick=0,
                chip=0,
                dribble=0,
            )
        else:
            if self.dribbled_distance < 0.8:
                target_oren = np.atan2(target_y - current_y, target_x - current_x)
                return calculate_robot_velocities(
                    game=game,
                    pid_oren=pid_oren,
                    pid_trans=pid_trans,
                    robot_id=self.robot_id,
                    target_coords=self.target_coords,
                    target_oren=target_oren,
                    dribbling=True,
                )
            else:
                # if just crossed the threshold, push the ball forward
                self.dribbled_distance = 0
                self.dribbling_cooldown = int(self.cooldown_sec * 60)
                self.last_point_with_ball = (current_x, current_y)
                return RobotCommand(
                    local_forward_vel=0,
                    local_left_vel=0,
                    angular_vel=0,
                    kick=0,
                    chip=0,
                    dribble=0,
                )

    def update_coord(self, next_coords: Tuple[int]):
        self.target_coords = next_coords

    def _update_dribble_distance(
        self, current_point: tuple[float, float], has_ball: bool
    ):
        """
        Update the distance dribbled by the robot with the ball.
        """

        if self.last_point_with_ball is not None:
            last_d = distance(self.last_point_with_ball, current_point)
            self.dribbled_distance += last_d
        self.last_point_with_ball = current_point
