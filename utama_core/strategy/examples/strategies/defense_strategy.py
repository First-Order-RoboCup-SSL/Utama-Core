from typing import Any

import py_trees
from py_trees.composites import Selector, Sequence

from utama_core.config.roles import Role
from utama_core.config.tactics import Tactic
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.skills.src.block import block_attacker
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.goalkeep import goalkeep
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


class FindBlockingTarget(AbstractBehaviour):
    def __init__(self):
        super().__init__(name="FindBlockingTarget2BB")

    def setup_(self):
        self.blackboard.register_key(
            key="blocking_target",
            access=py_trees.common.Access.WRITE,
            required=True,
        )

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        enemy, _ = game.proximity_lookup.closest_to_ball(TeamType.ENEMY)
        self.blackboard.blocking_target = enemy.id
        return py_trees.common.Status.SUCCESS


class BlockAttackerStep(AbstractBehaviour):
    """
    A behaviour that commands a robot to block the closest enemy robot to the ball.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot to check for ball possession. Typically from the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to block the attacker.
    """

    def setup_(self):
        self.blackboard.register_key(
            key="defender_id",
            access=py_trees.common.Access.READ,
            required=True,
        )
        self.blackboard.register_key(
            key="blocking_target",
            access=py_trees.common.Access.READ,
            required=True,
        )

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        robot_id = self.blackboard.defender_id
        blocking_target = self.blackboard.blocking_target
        command = block_attacker(
            game,
            self.blackboard.motion_controller,
            robot_id,
            blocking_target,
            True,
        )
        self.blackboard.cmd_map[robot_id] = command
        return py_trees.common.Status.RUNNING


class BlockPlay(AbstractBehaviour):
    """A behaviour that decides whether to play as attacker or defender based on the game state, and assigns a defender if defending."""

    def __init__(self):
        super().__init__(name="BlockPlay?")

    def update(self) -> py_trees.common.Status:
        tactic = self.blackboard.get("tactic")
        # I am skipping the play deciding logic for now but eventually it will check if we should be running a specific attacking or defending play.
        # example: if Play.ScoreGoal:
        if tactic == Tactic.DEFENDING:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class SetBlocker(AbstractBehaviour):
    """A behaviour that sets the defender robot"""

    def __init__(self):
        super().__init__(name="SetDefender2BB")

    def setup_(self):
        # Register the defender_id key in the blackboard
        self.blackboard.register_key(key="defender_id", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        for robot_id, role in self.blackboard.role_map.items():
            if role == Role.MIDFIELDER:
                self.blackboard.defender_id = robot_id
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SetRoles(AbstractBehaviour):
    """A behaviour that sets the roles of the robots."""

    def setup_(self):
        # Register the role_map key in the blackboard
        self.blackboard.register_key(key="role_map", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.role_map = {
            0: Role.MIDFIELDER,
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
        self.blackboard.tactic = Tactic.DEFENDING
        return py_trees.common.Status.SUCCESS


class DefenceStrategy(AbstractStrategy):
    def __init__(self):
        """Initializes the DefenceStrategy."""
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 3 <= n_runtime_friendly <= 5 and 1 <= n_runtime_enemy <= 6:
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
        else:
            return empty_command(True)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        root = Sequence(name="CoachRoot", memory=True)

        play_selector = Selector(name="PlayDecider", memory=False)

        # Play sequence for blocking the opponent
        block_play = Sequence(name="BlockPlaySequence", memory=True)

        # Play sequence for attacking the opponent
        attacking_play = Sequence(name="AttackingPlaySequence", memory=True)

        ### Assemble the tree ###

        root.add_children([SetTactics(), SetRoles(), play_selector])

        play_selector.add_children([block_play, attacking_play])

        block_play.add_child(BlockPlay())
        block_play.add_child(SetBlocker())
        block_play.add_child(FindBlockingTarget())
        block_play.add_child(BlockAttackerStep())

        return root
