import py_trees
from entities.game import Game, Robot
from entities.data.command import RobotCommand
from strategy.common import AbstractStrategy, AbstractBehaviour
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import HasBall
from strategy.common.roles import Role
from skills.src.go_to_ball import go_to_ball
from skills.src.defend_parameter import defend_parameter
from skills.src.goalkeep import goalkeep
from skills.src.utils.move_utils import empty_command


class GoToBallStep(AbstractBehaviour):
    """A behaviour that executes a single step of the go_to_ball skill."""

    def __init__(self, name="GoToBallStep"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        env = self.blackboard.rsim_env
        if env:
            v = game.friendly_robots[self.blackboard.robot_id].v
            p = game.friendly_robots[self.blackboard.robot_id].p
            env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")
        command = go_to_ball(
            game,
            self.blackboard.motion_controller,
            self.blackboard.robot_id,
        )
        print(f"GoToBallStep command {self.blackboard.robot_id}: {command}")
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING
    
class SetRoles(AbstractBehaviour):
    """A behaviour that sets the roles of the robots."""

    def __init__(self, name="SetRoles"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        self.blackboard.role_map = {
            0: Role.STRIKER,
            1: Role.DEFENDER,
            2: Role.GOALKEEPER,
        }
        return py_trees.common.Status.SUCCESS

class DefendStrategy(AbstractStrategy):
    def __init__(self, robot_id: int, opp_strategy: bool = False):
        """
        Initializes the DefendStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control to go to ball.
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
        """Factory function to create a complete go_to_ball behaviour tree."""

        # Create the root of the behaviour tree
        root = py_trees.composites.Sequence(name="CoachRoot", memory=True)

        # Create the SetRoles behaviour
        set_roles = SetRoles()
        
        # Root sequence for the whole behaviour
        go_to_ball = py_trees.composites.Sequence(name="GoToBall", memory=True)

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
        go_to_ball.add_child(set_robot_id)
        go_to_ball.add_child(has_ball_selector)
        
        root.add_child(set_roles)
        root.add_child(go_to_ball)

        return root
