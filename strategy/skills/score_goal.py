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
        """
        Aim first; once aimed, check goal-blocked.
        If blocked at that moment -> whole module FAILS.
        If clear -> kick.
        
        **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot that will perform the score goal action.
            - `best_shot` (float): The y-coordinate of the optimal shot target on the goal line. typically from the `ShouldScoreGoal` node.
            - `target_orientation` (float): The desired orientation angle in radians. typically from the `ShouldScoreGoal` node.

        """

        # 1) Aim until aligned: condition-first selector pattern
        aim_until_aligned = py_trees.composites.Selector(name="AimUntilAligned", memory=False)
        aim_until_aligned.add_children([
            OrenAtTargetThreshold(name="IsAimed?"),   # SUCCESS only when within tolerance
            TurnOnSpotStep(name="TurnToAim"),         # return RUNNING while turning; never SUCCESS
        ])

        # 2) Guard right before the kick: invert GoalBlocked so blocked => FAILURE
        goal_clear_guard = py_trees.decorators.Inverter(
            name="FailIfGoalBlockedNow",
            child=GoalBlocked(name="IsGoalBlocked?")
        )

        # 3) Kick
        kick = KickStep(name="Kick!")

        # Root: once aiming succeeds, check guard; if blocked now -> FAIL; else kick.
        root = py_trees.composites.Sequence(name="ShootModule", memory=False)
        root.add_children([
            aim_until_aligned,   # RUNNING until aimed; SUCCESS when aimed
            goal_clear_guard,    # if blocked now -> FAILURE (whole module fails)
            kick,                # executes only if goal is currently clear
        ])
        return root
