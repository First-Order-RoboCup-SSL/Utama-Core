import py_trees
from skills.src.utils.move_utils import turn_on_spot, kick
from skills.src.go_to_ball import go_to_ball
from strategy.common import AbstractBehaviour


class TurnOnSpotStep(AbstractBehaviour):
    """
    Executes a single command step to turn a robot on the spot.

    This behavior is an action that calls the `turn_on_spot` skill to generate
    a command for the specified robot. It writes this command to the blackboard
    and continuously returns `RUNNING` to ensure the robot keeps turning
    until interrupted by a higher-priority behavior.

    Blackboard Interaction:
        - `robot_id` (int): The ID of the robot to command. Usually set through the `SetBlackboardVariable` node.
        - `target_orientation` (float): The desired final orientation in radians.

    Returns:
        py_trees.common.Status.RUNNING: On every tick to continue the action.
    """
    def __init__(self, name="TurnOnSpotStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()

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
    """
    Executes a single, instantaneous kick command for a specified robot.

    This behavior is an action that issues a kick command. As kicking is
    considered an immediate action, this behavior generates the command,
    writes it to the blackboard, and returns SUCCESS in the same tick.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot that will perform the kick. Usually set through the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: Immediately after issuing the command.
    """
    def __init__(self, name="KickStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()

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
    """
    Executes a command step to move a robot towards the ball.

    This behavior is an action that calls the `go_to_ball` skill to generate
    a movement command. It writes this command to the blackboard and
    continuously returns RUNNING, allowing the robot to move towards the
    ball over multiple ticks.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot to command. Typically from the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: On every tick to continue the movement.
    """
    def __init__(self, name="GoToBallStep", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()

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