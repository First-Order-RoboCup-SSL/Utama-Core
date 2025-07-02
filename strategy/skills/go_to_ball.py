import py_trees
from py_trees.composites import Sequence, Selector
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import HasBall
from skills.src.go_to_ball import go_to_ball

from typing import Dict


class GoToBallStep(AbstractBehaviour):
    """A behaviour that executes a single step of the go_to_ball skill."""
    def __init__(self, name="GoToBallStep", remap_to: Dict[str, str] = None, opp_strategy: bool = False):
        super().__init__(name=name)
        self.remap_to = remap_to
        self.unique_key = (
            "Opponent" if opp_strategy else "My"
        )

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="robot_id",
            access=py_trees.common.Access.READ,
            remap_to=self.remap_to["robot_id"],
        )

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        env = self.blackboard.get(self.unique_key + "/rsim_env")
        if env:
            v = game.friendly_robots[self.blackboard.robot_id].v
            p = game.friendly_robots[self.blackboard.robot_id].p
            env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")
        command = go_to_ball(
            game,
            self.blackboard.get(self.unique_key + "/motion_controller"),
            self.blackboard.get("robot_id"),  # Use remapped robot_id
        )
        self.blackboard.cmd_map[self.blackboard.get("robot_id")] = command
        return py_trees.common.Status.RUNNING


class GoToBallStrategy(AbstractStrategy):
    def __init__(self, robot_id: int, opp_strategy: bool = False):
        """
        Initializes the GoToBallStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__(opp_strategy=opp_strategy)

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete go_to_ball behaviour tree."""

        unique_key = f"{'My' if not self.opp_strategy else 'Opponent'}"

        # Main logic for the robot
        go_to_ball_logic = Selector(
            name="GoToBallSelector",
            memory=False,
            children=[
                HasBall(remap_to={"robot_id": "/go_to_ball/robot_id"}, opp_strategy=self.opp_strategy),
                GoToBallStep(remap_to={"robot_id": "/go_to_ball/robot_id"}, opp_strategy=self.opp_strategy),
            ],
        )

        # Root of the tree that sets up the blackboard first
        root = Sequence(
            name="GoToBallRoot",
            memory=True, # Use memory to ensure setup runs only once
            children=[
                SetBlackboardVariable(
                    name="SetRobotID",
                    variable_name="robot_id", # Use a general name
                    value=self.robot_id,
                    remap_to={"robot_id": "/go_to_ball/robot_id"},
                    opp_strategy=self.opp_strategy
                ),
                go_to_ball_logic, # Run the main logic after setup
            ],
        )

        return root