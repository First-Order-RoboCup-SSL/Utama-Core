import py_trees
from strategy.common import AbstractStrategy, AbstractBehaviour
from skills.src.block import block_attacker
from entities.data.object import TeamType

class BlockAttackerStep(AbstractBehaviour):
    """A behaviour that executes a single step of the block_attacker skill."""
    def __init__(self, name="BlockAttackerStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()

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