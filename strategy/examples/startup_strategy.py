from typing import Callable, Tuple, Optional
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.data.command import RobotCommand
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID, get_grsim_pids
from robot_control.src.skills import face_ball, go_to_point
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.abstract_strategy import AbstractStrategy
import numpy as np
import logging

logger = logging.getLogger(__name__)

from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)


class StartupStrategy(AbstractStrategy):
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def step(self, present_future_game: PresentFutureGame):
        START_FORMATION = (
            RIGHT_START_ONE
            if present_future_game.current.my_team_is_right
            else LEFT_START_ONE
        )

        for robot_id, robot_data in present_future_game.current.friendly_robots.items():
            target_coords = START_FORMATION[robot_id]
            command = self._calculate_robot_velocities(
                robot_id, target_coords, present_future_game, face_ball=True
            )
            logger.info(command)
            self.robot_controller.add_robot_commands(command, robot_id)
        self.robot_controller.send_robot_commands()

    def _calculate_robot_velocities(
        self,
        robot_id: int,
        target_coords: Tuple[float, float],
        present_future_game: PresentFutureGame,
        face_ball=False,
    ) -> RobotCommand:

        ball_p = present_future_game.current.ball.p
        current_p = present_future_game.current.friendly_robots[robot_id].p
        target_oren = (ball_p.like(current_p) - current_p).phi if face_ball else None
        return go_to_point(
            present_future_game.current,
            self.pid_oren,
            self.pid_trans,
            robot_id,
            target_coords,
            target_oren,
        )
