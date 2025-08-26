import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Inverter
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import GoalScored
from strategy.utils.atk_utils import ShouldScoreGoal
from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.skills.score_goal import ScoreGoalStrategy
from strategy.skills.dribble import DribbleStrategy
from strategy.skills.block_attacker import BlockAttackerStep
from py_trees.composites import Selector, Sequence

from config.roles import Role
from config.tactics import Tactic
from entities.data.object import TeamType
from entities.game import Game
from skills.src.defend_parameter import defend_parameter
from skills.src.goalkeep import goalkeep
from skills.src.utils.move_utils import empty_command
from strategy.common import AbstractBehaviour, AbstractStrategy
from strategy.skills.block_attacker import BlockAttackerStep
from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.skills.score_goal import ScoreGoalStrategy
from strategy.utils.atk_utils import ShouldScoreGoal
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import GoalScored


class SetRoles(AbstractBehaviour):
    """A behaviour that sets the roles of the robots."""

    def setup_(self):
        # Register the role_map key in the blackboard
        self.blackboard.register_key(key="role_map", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.role_map = {
            0: Role.STRIKER,
            1: Role.DEFENDER,
            2: Role.GOALKEEPER,
        }
        return py_trees.common.Status.SUCCESS


class SetTactics(AbstractBehaviour):
    """A behaviour that sets the tactics of the robots."""

    def setup_(self):
        # Register the tactic key in the blackboard
        self.blackboard.register_key(key="tactic", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        team, _ = game.proximity_lookup.closest_to_ball()

        if team.team_type == TeamType.FRIENDLY:
            self.blackboard.tactic = Tactic.ATTACKING
        else:
            self.blackboard.tactic = Tactic.DEFENDING

        return py_trees.common.Status.SUCCESS


class ScoreGoalPlay(AbstractBehaviour):
    """A behaviour that decides whether to play as attacker or defender based on the game state."""

    def __init__(self, name: str = "ScoreGoalPlay?"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        tactic = self.blackboard.get("tactic")
        # I am skipping the play deciding logic for now but eventually it will check if we should be running a specific attacking or defending play.
        # example: if Play.ScoreGoal:
        if tactic == Tactic.ATTACKING:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class BlockAttackerPlay(AbstractBehaviour):
    """A behaviour that blocks the attacker play."""

    def __init__(self, name="BlockAttackerPlay?"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        tactic = self.blackboard.get("tactic")
        # I am skipping the play deciding logic for now but eventually it will check if we should be running a specific attacking or defending play.
        if tactic == Tactic.DEFENDING:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class DemoStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the DemoStrategy with a specific robot ID.

        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def execute_default_action(self, game: Game, role: Role, robot_id: int):
        """
        Called by StrategyRunner: Execute the default action for the robot.
        This is used when no specific command is set in the blackboard after the coach tree for this robot.
        """
        if role == Role.DEFENDER:
            return defend_parameter(game, self.blackboard.motion_controller, robot_id)
        elif role == Role.GOALKEEPER:
            return goalkeep(game, self.blackboard.motion_controller, robot_id)
        elif role == Role.STRIKER:
            return empty_command(True)
        else:
            return empty_command(True)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete score_goal behaviour tree."""

        # 1. A sequence for the main robot action: acquire the ball, then score or dribble.
        score_goal_branch = ScoreGoalStrategy(self.robot_id).create_module()
        dribble_branch = DribbleStrategy(self.robot_id).create_module()

        acquire_then_play = Sequence(
            name="AcquireThenPlay",
            memory=True,
            children=[
                GoToBallStrategy(self.robot_id).create_module(),
                Selector(
                    name="ScoreOrDribble",
                    memory=False,
                    children=[
                        Sequence(
                            name="ScoreGoalSequence",
                            memory=False,
                            children=[
                                ShouldScoreGoal(name="ShouldScoreGoal?"),
                                score_goal_branch,
                            ]
                        ),
                        dribble_branch,
                    ],
                ),
            ],
        )

        # 2. A top-level selector that stops the robot if a goal has been scored.
        # This prevents the robot from acting unnecessarily.
        goal_scored_selector = Selector(name="StopIfGoalScored", memory=True)
        goal_scored_selector.add_children(
            [GoalScored(name="IsGoalScored?"), acquire_then_play]
        )

        # 3. The root of the tree, which initializes and then runs the main logic.
        attacking = Sequence(name="ScoreGoal", memory=False)
        set_atk_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name="robot_id",
            value=self.robot_id,
        )
        attacking.add_children(
            [
                ScoreGoalPlay(name="ScoreGoalPlay?"),
                set_atk_id,
                goal_scored_selector,
            ]
        )

        defending = Sequence(name="Block", memory=False)
        set_def_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name="robot_id",
            value=self.robot_id,
        )
        defending.add_children(
            [
                BlockAttackerPlay(name="BlockAttackerPlay?"),
                set_def_id,
                BlockAttackerStep(name="BlockAttackerStep"),
            ]
        )

        play_selector = Selector(name="PlayDecider", memory=False)
        play_selector.add_children(
            [
                attacking,
                defending,
            ]
        )

        # Create the root of the behaviour tree
        coach_root = py_trees.composites.Sequence(name="CoachRoot", memory=False)
        coach_root.add_children([SetRoles(), SetTactics(), play_selector])

        return coach_root
