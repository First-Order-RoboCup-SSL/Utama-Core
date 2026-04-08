import py_trees
from py_trees.composites import Sequence

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.formations import FormationType, get_formations
from utama_core.config.physical_constants import MAX_ROBOTS
from utama_core.entities.data.vector import Vector2D
from utama_core.global_utils.math_utils import compute_bounding_zone_from_points
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


def generate_starting_positions(is_right_team: bool):
    """
    Generate starting and target formations based on team side.
    """
    left_formation, right_formation = get_formations(
        STANDARD_FIELD_DIMS.full_field_bounds,
        MAX_ROBOTS,
        MAX_ROBOTS,
        formation_type=FormationType.START_ONE,
    )
    start_formation = right_formation if is_right_team else left_formation
    target_formation = start_formation.copy()
    target_formation.reverse()
    return start_formation, target_formation


class StartupFormationStep(AbstractBehaviour):
    """
    A behaviour that commands all robots to move to their starting formation positions.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to move.
    """

    def initialise(self):
        _, self.target_formation = generate_starting_positions(self.blackboard.game.my_team_is_right)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        for robot_id in game.friendly_robots.keys():
            tx, ty, _ = self.target_formation[robot_id]
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
        return True  # No specific robot count requirements

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_req(self):
        return STANDARD_FIELD_DIMS.full_field_bounds  # enforce full field required

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        coach_root = Sequence(name="CoachRoot", memory=False)

        ### Assemble the tree ###

        coach_root.add_children([StartupFormationStep()])

        return coach_root
