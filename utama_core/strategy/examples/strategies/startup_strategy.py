import logging
from typing import Tuple

import py_trees

from utama_core.config.defaults import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy

logger = logging.getLogger(__name__)


class StartupFormationStep(AbstractBehaviour):
    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        motion_controller = self.blackboard.motion_controller

        start_formation = RIGHT_START_ONE if game.my_team_is_right else LEFT_START_ONE

        for robot_id in game.friendly_robots.keys():
            tx, ty, _ = start_formation[robot_id]
            command = go_to_point(
                game,
                motion_controller,
                robot_id,
                Vector2D(tx, ty),
                False,
            )
            self.blackboard.cmd_map[robot_id] = command

        return py_trees.common.Status.RUNNING


class StartupStrategy(AbstractStrategy):
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if n_runtime_friendly == 6:
            return True
        return False

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        coach_root = py_trees.composites.Sequence(name="CoachRoot", memory=False)
        coach_root.add_children([StartupFormationStep()])
        return coach_root
