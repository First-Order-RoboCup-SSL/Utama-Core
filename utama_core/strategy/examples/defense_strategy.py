from typing import Optional

import py_trees
from py_trees.composites import Selector, Sequence

from utama_core.config.enums import Role, Tactic
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.skills.src.block import block_attacker
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.goalkeep import goalkeep
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


class FindBlockingTarget(AbstractBehaviour):
    def __init__(self, wr_blocking_target: str, name: Optional[str] = "FindBlockingTarget2BB"):
        super().__init__(name=name)
        self.blocking_target_key = wr_blocking_target

    def setup_(self):
        self.blackboard.register_key(
            key=self.blocking_target_key,
            access=py_trees.common.Access.WRITE,
            required=True,
        )

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        enemy, _ = game.proximity_lookup.closest_to_ball(TeamType.ENEMY)
        self.blackboard.set(self.blocking_target_key, enemy.id)
        return py_trees.common.Status.SUCCESS


class BlockAttackerStep(AbstractBehaviour):
    """
    A behaviour that commands a robot to block the closest enemy robot to the ball.

    **Blackboard Interaction:**
        Reads:
            - `rd_defender_id` (int): The ID of the robot to check for ball possession. Typically from the `SetBlackboardVariable` node.
            - `rd_blocking_target` (int): The ID of the enemy robot to block. Typically from the `FindBlockingTarget` node.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to block the attacker.
    """

    def __init__(
        self,
        rd_defender_id: str,
        rd_locking_target: str,
        name: Optional[str] = "BlockAttackerStep",
    ):
        super().__init__(name=name)
        self.defender_id_key = rd_defender_id
        self.locking_target_key = rd_locking_target

    def setup_(self):
        self.blackboard.register_key(
            key=self.defender_id_key,
            access=py_trees.common.Access.READ,
            required=True,
        )
        self.blackboard.register_key(
            key=self.locking_target_key,
            access=py_trees.common.Access.READ,
            required=True,
        )

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        robot_id = self.blackboard.get(self.defender_id_key)
        blocking_target = self.blackboard.get(self.locking_target_key)
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
    """
    A behaviour that decides whether to play as attacker or defender based on the game state, and assigns a defender if defending.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The team is in defending mode.
        - `py_trees.common.Status.FAILURE`: The team is not in defending mode.
    """

    def __init__(self, rd_tactic: str = "tactic"):
        super().__init__(name="BlockPlay?")
        self.tactic_key = rd_tactic

    def update(self) -> py_trees.common.Status:
        tactic = self.blackboard.get(self.tactic_key)
        # I am skipping the play deciding logic for now but eventually it will check if we should be running a specific attacking or defending play.
        # example: if Play.ScoreGoal:
        if tactic == Tactic.DEFENDING:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class SetBlocker(AbstractBehaviour):
    """
    A behaviour that sets the defender robot

    **Blackboard Interaction:**
        - Writes:
            - `rd_defender_id` (int): The ID of the robot assigned as defender.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The defender ID has been successfully set.
        - `py_trees.common.Status.FAILURE`: No suitable robot found to assign as defender
    """

    def __init__(self, wr_defender_id: str):
        super().__init__(name="SetDefender2BB")
        self.defender_id_key = wr_defender_id

    def setup_(self):
        # Register the defender_id key in the blackboard
        self.blackboard.register_key(key=self.defender_id_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        for robot_id, role in self.blackboard.role_map.items():
            if role == Role.MIDFIELDER:
                self.blackboard.set(self.defender_id_key, robot_id)
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SetRoles(AbstractBehaviour):
    """
    A behaviour that sets the roles of the robots.

    **Blackboard Interaction:**
        - Writes:
            - `role_map` (dict): A mapping of robot IDs to their assigned roles.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The role map has been successfully set.
    """

    def __init__(self, wr_role_map: str = "role_map", name: Optional[str] = "SetRoles2BB"):
        super().__init__(name=name)
        self.role_map_key = wr_role_map

    def setup_(self):
        # Register the role_map key in the blackboard
        self.blackboard.register_key(key=self.role_map_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(
            self.role_map_key,
            {
                0: Role.MIDFIELDER,
                1: Role.DEFENDER,
                2: Role.GOALKEEPER,
            },
        )
        return py_trees.common.Status.SUCCESS


class SetTactics(AbstractBehaviour):
    """
    A behaviour that sets the tactics of the robots.

    **Blackboard Interaction:**
        - Writes:
            - `tactic` (Tactic): The tactic to be set for the team.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The tactic has been successfully set.
    """

    def __init__(self, wr_tactic: str = "tactic", name: Optional[str] = "SetTactics2BB"):
        super().__init__(name=name)
        self.tactic_key = wr_tactic

    def setup_(self):
        # Register the tactic key in the blackboard
        self.blackboard.register_key(key=self.tactic_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(self.tactic_key, Tactic.DEFENDING)
        return py_trees.common.Status.SUCCESS


class DefenceStrategy(AbstractStrategy):
    def __init__(self):
        """Initializes the DefenceStrategy."""
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 3 <= n_runtime_friendly <= 6 and 1 <= n_runtime_enemy <= 6:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line, includes_opp_goal_line):
        return True  # No specific goal line requirements

    def get_min_bounding_zone(self):
        return None  # No specific bounding zone requirements

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

        defender_id_key = "defender_id"
        blocking_target_key = "locking_target"

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
        block_play.add_child(SetBlocker(wr_defender_id=defender_id_key))
        block_play.add_child(FindBlockingTarget(wr_blocking_target=blocking_target_key))
        block_play.add_child(BlockAttackerStep(rd_defender_id=defender_id_key, rd_locking_target=blocking_target_key))

        return root
