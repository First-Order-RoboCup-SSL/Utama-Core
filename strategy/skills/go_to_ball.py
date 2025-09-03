import py_trees
from py_trees.composites import Selector, Sequence

from strategy.common import AbstractStrategy
from strategy.utils.action_nodes import GoToBallStep
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import HasBall


class GoToBallStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the GoToBallStrategy with a specific robot ID.

        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete go_to_ball behaviour tree."""

        # Main logic for the robot
        go_to_ball_logic = Selector(
            name="GoToBallSelector",
            memory=False,
            children=[
                HasBall(),
                GoToBallStep(),
            ],
        )

        # Root of the tree that sets up the blackboard first
        root = Sequence(
            name="GoToBallModule",
            memory=True,  # Use memory to ensure setup runs only once
            children=[
                SetBlackboardVariable(
                    name="SetRobotID",
                    variable_name="robot_id",  # Use a general name
                    value=self.robot_id,
                ),
                go_to_ball_logic,  # Run the main logic after setup
            ],
        )

        return root

    def create_module(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete go_to_ball behaviour tree."""

        # Main logic for the robot
        go_to_ball_logic = Selector(
            name="GoToBallSelector",
            memory=False,
            children=[
                HasBall(),
                GoToBallStep(),
            ],
        )

        return go_to_ball_logic
