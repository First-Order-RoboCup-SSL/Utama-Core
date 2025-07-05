import py_trees
from skills.src.utils.move_utils import turn_on_spot, kick
from skills.src.go_to_ball import go_to_ball
from strategy.common import AbstractBehaviour


class TurnOnSpotStep(AbstractBehaviour):
    """A behaviour that executes a single step of the turn_on_spot skill."""
    def __init__(self, name="TurnOnSpotStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)

        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_orientation", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        # print(f"Executing TurnOnSpotStep for robot {self.blackboard.robot_id}, target orientation: {self.blackboard.target_orientation}")
        game = self.blackboard.game.current
        env = self.blackboard.rsim_env
        if env:
            pass
        command = turn_on_spot(
            game,
            self.blackboard.motion_controller,
            self.blackboard.robot_id,  # Use remapped robot_id
            self.blackboard.target_orientation,  # Use target orientation from blackboard
            True,
        )
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING

class KickStep(AbstractBehaviour):
    """A behaviour that executes a single step of the kick skill."""
    def __init__(self, name="KickStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)

        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_orientation", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        # print(f"Executing KickStep for robot {self.blackboard.robot_id}")
        env = self.blackboard.rsim_env
        if env:
            pass
        command = kick()
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.SUCCESS
    
class GoToBallStep(AbstractBehaviour):
    """A behaviour that executes a single step of the go_to_ball skill."""
    def __init__(self, name="GoToBallStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)

        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        # print(f"Executing GoToBallStep for robot {self.blackboard.robot_id}")
        game = self.blackboard.game.current
        env = self.blackboard.rsim_env
        if env:
            v = game.friendly_robots[self.blackboard.robot_id].v
            p = game.friendly_robots[self.blackboard.robot_id].p
            env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")
            
        command = go_to_ball(
            game,
            self.blackboard.motion_controller,
            self.blackboard.robot_id,  # Use remapped robot_id
        )
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.RUNNING