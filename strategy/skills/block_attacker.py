import py_trees

from entities.data.object import TeamType
from skills.src.block import block_attacker
from strategy.common import AbstractBehaviour


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
