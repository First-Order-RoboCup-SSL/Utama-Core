import numpy as np
from typing import Tuple
from time import sleep

from entities.game import Game
from entities.data.command import RobotCommand, RobotInfo
from entities.data.vision import BallData, RobotData

from motion_planning.src.pid import PID
from global_utils.math_utils import distance

from robot_control.src.utils.motion_planning_utils import calculate_robot_velocities
from robot_control.src.skills import go_to_ball, go_to_point


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
        pid_oren: PID,
        pid_trans: PID,
        game: Game,
        robot_id: int,
        target_coords: Tuple[float, float],
        dribble_speed: float = 3.0,
        tolerance: float = 0.1,
        augment: bool = False,
    ):
        self.pid_oren = pid_oren
        self.pid_trans = pid_trans
        self.game = game
        self.robot_id = robot_id
        self.target_coords = target_coords
        self.dribble_speed = dribble_speed
        self.tolerance = tolerance

        self.max_dribble_d = 0.8
        self.last_point_with_ball = None
        self.dribbled_distance = 0  # TODO: this needs to be stored as a global robot state in the future (because it is accessed by all functions involving movement)
        self.paused = False
        self.augment = augment

    def enact(self, has_ball: bool):
        this_robot_data = self.game.friendly_robots[self.robot_id].robot_data
        current_x, current_y, current_oren = this_robot_data
        target_x, target_y = self.target_coords
        self._update_dribble_distance(
            (current_x, current_y), has_ball
        )  # update distance traveled with ball

        if distance((current_x, current_y), self.target_coords) <= self.tolerance:
            return RobotCommand(
                local_forward_vel=0,
                local_left_vel=0,
                angular_vel=0,
                kick=0,
                chip=0,
                dribble=0,
            )
        else:
            if not has_ball:
                ball_data = self.game.ball
                if self.augment:
                    delta_x = self.game.ball.x - current_x
                    delta_y = self.game.ball.y - current_y
                    ball_data = BallData(0, 
                        delta_x * 6 + current_x, delta_y * 6 + current_y, 0
                    )
                return go_to_ball(
                    self.pid_oren,
                    self.pid_trans,
                    this_robot_data,
                    self.robot_id,
                    ball_data,
                )
            elif self.dribbled_distance < 0.65:
                target_oren = np.atan2(target_y - current_y, target_x - current_x)
                return calculate_robot_velocities(
                    self.pid_oren,
                    self.pid_trans,
                    this_robot_data,
                    self.robot_id,
                    self.target_coords,
                    target_oren,
                    dribbling=True,
                )
            else:
                # if just crossed the threshold, push the ball forward
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
        if not has_ball:
            self.last_point_with_ball = None
            self.dribbled_distance = 0
        else:
            if self.last_point_with_ball is not None:
                last_d = distance(self.last_point_with_ball, current_point)
                self.dribbled_distance += last_d
            self.last_point_with_ball = current_point
