import py_trees
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import GoalScored
from skills.src.score_goal import score_goal


class ScoreGoalStep(AbstractBehaviour):
    """A behaviour that executes a single step of the score_goal skill."""

    def __init__(self, name="GoToBallStep"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current

        try:
            env = self.blackboard.rsim_env
            v = game.friendly_robots[self.blackboard.robot_id].v
            p = game.friendly_robots[self.blackboard.robot_id].p
            env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")

            command = score_goal(
                game, self.blackboard.motion_controller, self.blackboard.robot_id, env
            )
        except:
            command = score_goal(
                game,
                self.blackboard.motion_controller,
                self.blackboard.robot_id,
            )

        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING


class ScoreGoalStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """
        Initializes the ScoreGoalStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete score_goal behaviour tree."""

        # Root sequence for the whole behaviour
        root = py_trees.composites.Sequence(name="ScoreGoal", memory=True)

        # A child sequence to set the robot_id on the blackboard
        set_robot_id = SetBlackboardVariable(
            name="SetTargetRobotID", variable_name="robot_id", value=self.robot_id
        )

        # A selector to decide whether to get the ball or stop
        goal_scored_selector = py_trees.composites.Selector(
            name="GoalDScoredSelector", memory=False
        )
        goal_scored_selector.add_child(GoalScored())
        goal_scored_selector.add_child(ScoreGoalStep())

        # Assemble the tree
        root.add_child(set_robot_id)
        root.add_child(goal_scored_selector)

        return root
