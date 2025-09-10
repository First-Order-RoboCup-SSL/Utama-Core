from typing import Any

import py_trees
from py_trees.composites import Selector, Sequence

from utama_core.config.roles import Role
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.skills.src.block import block_attacker
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.goalkeep import goalkeep
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


class BlockAttackerStep(AbstractBehaviour):
    """A behaviour that executes a single step of the block_attacker skill.

    Expects a "robot_id" key in the blackboard to identify which robot to control.
    """

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        # print(f"Executing BlockAttackerStep for robot {self.blackboard.robot_id}")
        game = self.blackboard.game.current
        enemy, _ = game.proximity_lookup.closest_to_ball(TeamType.ENEMY)

        command = block_attacker(
            game,
            self.blackboard.motion_controller,
            self.blackboard.robot_id,  # Use remapped robot_id
            enemy.id,  # Use the closest enemy robot to the ball
            True,
        )
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING


class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any, opp_strategy: bool = False):
        super().__init__(name=name)
        # Store the configuration, but DO NOT use the blackboard here.
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        # print(f"Setting {self.variable_name} to {self.value} on the blackboard.")
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS


class SetRoles(AbstractBehaviour):
    """A behaviour that sets the roles of the robots."""

    def setup_(self):
        # Register the role_map key in the blackboard
        self.blackboard.register_key(key="role_map", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.role_map = {
            0: Role.DEFENDER,
            1: Role.DEFENDER,
            2: Role.GOALKEEPER,
        }
        return py_trees.common.Status.SUCCESS


class DefenceStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the DefendStrategy with a specific robot ID.

        :param robot_id: The ID of the robot this strategy will control to go to ball.
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
        """Factory function to create a complete go_to_ball behaviour tree."""

        # Create the root of the behaviour tree
        root = Sequence(name="CoachRoot", memory=True)

        # Create the SetRoles behaviour
        set_roles = SetRoles()

        # Root sequence for the whole behaviour
        block = Sequence(name="GoToBall", memory=True)

        # A child sequence to set the robot_id on the blackboard
        set_robot_id = SetBlackboardVariable(name="SetTargetRobotID", variable_name="robot_id", value=self.robot_id)

        block.add_child(set_robot_id)
        block.add_child(BlockAttackerStep())

        root.add_child(set_roles)
        root.add_child(block)

        return root
