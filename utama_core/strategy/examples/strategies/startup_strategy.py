import py_trees

from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


class StartupFormationStep(AbstractBehaviour):
    """
    A behaviour that commands all robots to move to their starting formation positions.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to move.
    """

    def initialise(self):
        self.start_formation = RIGHT_START_ONE if self.blackboard.game.my_team_is_right else LEFT_START_ONE
        self.start_formation.reverse()

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        for robot_id in game.friendly_robots.keys():
            tx, ty, _ = self.start_formation[robot_id]
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
        """Factory function to create a complete behaviour tree."""
        coach_root = py_trees.composites.Sequence(name="CoachRoot", memory=False)
        coach_root.add_children([StartupFormationStep()])
        return coach_root
