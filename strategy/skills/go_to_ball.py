import py_trees
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import HasBall
from skills.src.go_to_ball import go_to_ball


class GoToBallStep(AbstractBehaviour):
    """A behaviour that executes a single step of the go_to_ball skill."""

    def __init__(self, name="GoToBallStep"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        command = go_to_ball(
            self.blackboard.present_future_game.current,
            self.blackboard.pid_oren,
            self.blackboard.pid_trans,
            self.blackboard.robot_id,
        )
        self.blackboard.robot_controller.add_robot_commands(
            command, self.blackboard.robot_id
        )
        return py_trees.common.Status.RUNNING


class GoToBallStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """
        Initializes the GoToBallStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def create_behaviour(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete go_to_ball behaviour tree."""

        # Root sequence for the whole behaviour
        root = py_trees.composites.Sequence(name="GoToBall", memory=True)

        # A child sequence to set the robot_id on the blackboard
        set_robot_id = SetBlackboardVariable(
            name="SetTargetRobotID", variable_name="robot_id", value=self.robot_id
        )

        # A selector to decide whether to get the ball or stop
        has_ball_selector = py_trees.composites.Selector(
            name="HasBallSelector", memory=False
        )
        has_ball_selector.add_child(HasBall())
        has_ball_selector.add_child(GoToBallStep())

        # Assemble the tree
        root.add_child(set_robot_id)
        root.add_child(has_ball_selector)

        return root


# TODO: add visualisation of the target point
# if self.env:
#     v = game.friendly_robots[self.target_id].v
#     p = game.friendly_robots[self.target_id].p
#     self.env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")
