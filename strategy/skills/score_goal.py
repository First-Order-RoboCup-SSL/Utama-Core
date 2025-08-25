import py_trees
from py_trees.composites import Selector, Sequence
from py_trees.decorators import Inverter

from skills.src.score_goal import score_goal
from strategy.common import AbstractBehaviour, AbstractStrategy
from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.utils.action_nodes import KickStep, TurnOnSpotStep
from strategy.utils.atk_utils import GoalBlocked, OrenAtTargetThreshold, ShouldScoreGoal
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import GoalScored

# Alternative option to using the ScoreGoalStrategy.create_module
class ScoreGoalStep(AbstractBehaviour):
    """A behaviour that executes a single step of the score_goal skill."""

    def __init__(self, name="ScoreGoalStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current

        if self.blackboard.rsim_env is not None:
            env = self.blackboard.rsim_env
            v = game.friendly_robots[self.blackboard.robot_id].v
            p = game.friendly_robots[self.blackboard.robot_id].p
            env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")

            command = score_goal(game, self.blackboard.motion_controller, self.blackboard.robot_id, env)
        else:
            command = score_goal(
                game,
                self.blackboard.motion_controller,
                self.blackboard.robot_id,
            )

        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING


class ScoreGoalStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the ScoreGoalStrategy with a specific robot ID.

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
        root = Sequence(name="ScoreGoal", memory=True)

        # A child sequence to set the robot_id on the blackboard
        set_robot_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name="robot_id",
            value=self.robot_id,
        )

        # A selector to decide whether to get the ball or stop
        goal_scored_selector = Selector(name="GoalDScoredSelector", memory=False)
        goal_scored_selector.add_child(GoalScored())

        # 1. A sequence for the main robot action: get the ball, then turn/aim/shoot.
        get_ball_and_shoot = Sequence(name="GetBallAndShoot", memory=False)
        go_to_ball_branch = GoToBallStrategy(self.robot_id).create_module()
        score_goal = self.create_module()
        get_ball_and_shoot.add_children(
            [
                go_to_ball_branch,
                ShouldScoreGoal(name="ShouldScoreGoal?"),
                score_goal,
            ]
        )
        goal_scored_selector.add_child(get_ball_and_shoot)

        # Assemble the tree
        root.add_child(set_robot_id)
        root.add_child(goal_scored_selector)

        return root

    def create_module(self) -> py_trees.behaviour.Behaviour:
        """Create a module for this strategy. This is used to create a module that can be used in other strategies.

        **Blackboard Interaction:**
            Reads:
                - `robot_id` (int): The ID of the robot that will perform the score goal action.
                - `best_shot` (float): The y-coordinate of the optimal shot target on the goal line. typically from the `ShouldScoreGoal` node.
                - `target_orientation` (float): The desired orientation angle in radians. typically from the `ShouldScoreGoal` node.
        """
        # 1. A sequence that attempts to kick only if all conditions are perfect.
        # It needs to be re-evaluated every tick, so memory=False.
        attempt_kick = Sequence(name="AttemptKick", memory=False)
        attempt_kick.add_children(
            [
                Inverter(
                    name="IsGoalNotBlocked?",
                    child=GoalBlocked(name="IsGoalBlocked?"),
                ),
                OrenAtTargetThreshold(name="IsAimed?"),
                KickStep(name="Kick!"),
            ]
        )

        # 2. A selector that decides whether to kick (if ready) or to turn and aim.
        # This is the core "aim and shoot" logic.
        aim_and_shoot = Selector(name="AimAndShoot", memory=False)
        aim_and_shoot.add_children(
            [
                attempt_kick,
                TurnOnSpotStep(name="TurnToAim"),
            ]
        )

        return aim_and_shoot
