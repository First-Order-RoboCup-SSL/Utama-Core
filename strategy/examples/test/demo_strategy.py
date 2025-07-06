import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Inverter
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import GoalScored
from strategy.utils.atk_utils import OrenAtTargetThreshold, GoalBlocked, ShouldScoreGoal
from strategy.utils.action_nodes import TurnOnSpotStep, KickStep
from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.skills.score_goal import ScoreGoalStrategy
from strategy.skills.block_attacker import BlockAttackerStep
from entities.data.object import TeamType
from strategy.common.roles import Role
from strategy.common.tactics import Tactic
from skills.src.defend_parameter import defend_parameter
from skills.src.goalkeep import goalkeep
from skills.src.utils.move_utils import empty_command
from entities.game import Game

class SetRoles(AbstractBehaviour):
    """A behaviour that sets the roles of the robots."""

    def __init__(self, name="SetRoles", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()
        
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

    def __init__(self, name="SetTactics", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()
        
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

    def __init__(self, name="ScoreGoalPlay?", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)
        
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

    def __init__(self, name="BlockAttackerPlay?", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)
        
    def update(self) -> py_trees.common.Status:
        tactic = self.blackboard.get("tactic")
        # I am skipping the play deciding logic for now but eventually it will check if we should be running a specific attacking or defending play.
        if tactic == Tactic.DEFENDING:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class DemoStrategy(AbstractStrategy):
    def __init__(self, robot_id: int, opp_strategy: bool = False):
        """
        Initializes the DemoStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__(opp_strategy=opp_strategy)

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False

    def execute_default_action(
        self, game: Game, role: Role, robot_id: int
    ):
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

        # 1. A sequence for the main robot action: get the ball, then turn/aim/shoot.
        get_ball_and_shoot = Sequence(name="GetBallAndShoot", memory=False)
        go_to_ball_branch = GoToBallStrategy(self.robot_id, self.opp_strategy).create_module()
        score_goal_branch = ScoreGoalStrategy(self.robot_id, self.opp_strategy).create_module()
        get_ball_and_shoot.add_children([
            go_to_ball_branch,
            ShouldScoreGoal(name="ShouldScoreGoal?", opp_strategy=self.opp_strategy),
            score_goal_branch
        ])

        # 2. A top-level selector that stops the robot if a goal has been scored.
        # This prevents the robot from acting unnecessarily.
        goal_scored_selector = Selector(name="StopIfGoalScored", memory=False)
        goal_scored_selector.add_children([
            GoalScored(name="IsGoalScored?", opp_strategy=self.opp_strategy),
            get_ball_and_shoot
        ])

        # 3. The root of the tree, which initializes and then runs the main logic.
        attacking = Sequence(name="ScoreGoal", memory=False)
        set_atk_id = SetBlackboardVariable(
                name="SetTargetRobotID", variable_name="robot_id", value=self.robot_id, opp_strategy=self.opp_strategy
            )
        attacking.add_children([
            ScoreGoalPlay(name="ScoreGoalPlay?", opp_strategy=self.opp_strategy),
            set_atk_id,
            goal_scored_selector
        ])
        
        defending = Sequence(name="Block", memory=False)
        set_def_id = SetBlackboardVariable(
                name="SetTargetRobotID", variable_name="robot_id", value=self.robot_id, opp_strategy=self.opp_strategy
            )
        defending.add_children([
            BlockAttackerPlay(name="BlockAttackerPlay?", opp_strategy=self.opp_strategy),
            set_def_id,
            BlockAttackerStep(name="BlockAttackerStep", opp_strategy=self.opp_strategy),
        ])
        
        play_selector = Selector(name="PlayDecider", memory=False)
        play_selector.add_children([
            attacking,
            defending,
        ])

        # Create the root of the behaviour tree
        coach_root = py_trees.composites.Sequence(name="CoachRoot", memory=False)

        # Create the SetRoles behaviour
        set_roles = SetRoles(opp_strategy=self.opp_strategy)
        set_tactics = SetTactics(opp_strategy=self.opp_strategy)

        coach_root.add_children([
            set_roles,
            set_tactics,
            play_selector
        ])

        return coach_root