from typing import Tuple
from config.defaults import LEFT_START_ONE, RIGHT_START_ONE
from entities.data.command import RobotCommand
from entities.game import Game
from skills.src.skills import go_to_point
from strategy.abstract_strategy import AbstractStrategy

import logging

logger = logging.getLogger(__name__)


class StartupStrategy(AbstractStrategy):
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def step(self, game: Game):
        START_FORMATION = (
            RIGHT_START_ONE if game.current.my_team_is_right else LEFT_START_ONE
        )

        for robot_id, robot_data in game.current.friendly_robots.items():
            target_coords = START_FORMATION[robot_id]
            command = self._move(robot_id, target_coords, game, face_ball=True)
            logger.info(command)
            self.robot_controller.add_robot_commands(command, robot_id)
        self.robot_controller.send_robot_commands()

    def _move(
        self,
        robot_id: int,
        target_coords: Tuple[float, float],
        game: Game,
        face_ball=False,
    ) -> RobotCommand:

        ball_p = game.current.ball.p
        current_p = game.current.friendly_robots[robot_id].p
        target_oren = (ball_p.like(current_p) - current_p).phi if face_ball else None
        return go_to_point(
            game.current,
            self.pid_oren,
            self.pid_trans,
            robot_id,
            target_coords,
            target_oren,
        )
