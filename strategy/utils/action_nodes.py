import py_trees
from skills.src.utils.move_utils import (
    turn_on_spot,
    kick,
    move,
    empty_command,
)
from skills.src.go_to_ball import go_to_ball
from skills.src.utils.move_utils import kick, turn_on_spot
from strategy.common import AbstractBehaviour
from entities.data.vector import Vector2D


class TurnOnSpotStep(AbstractBehaviour):
    """Executes a single command step to turn a robot on the spot.

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

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_orientation", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        print(f"Executing TurnOnSpotStep for robot {self.blackboard.robot_id}, target orientation: {self.blackboard.target_orientation}")
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
    """Executes a single, instantaneous kick command for a specified robot.

    This behavior is an action that issues a kick command. As kicking is
    considered an immediate action, this behavior generates the command,
    writes it to the blackboard, and returns SUCCESS in the same tick.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot that will perform the kick. Usually set through the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: Immediately after issuing the command.
    """

    def setup_(self):
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
    """Executes a command step to move a robot towards the ball.

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

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        print(f"Executing GoToBallStep for robot {self.blackboard.robot_id}")
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


class UpdateDribbleDistance(AbstractBehaviour):
    """Updates the distance a robot has dribbled the ball."""

    def __init__(self, name: str = "UpdateDribbleDistance"):
        super().__init__(name=name)
        self.last_point_with_ball: Vector2D = None

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="dribbled_distance", access=py_trees.common.Access.WRITE
        )
        self.blackboard.set(
            "dribbled_distance", 0.0, overwrite=True
        )


    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        robot = game.friendly_robots[self.blackboard.robot_id]
        current_point = Vector2D(robot.p.x, robot.p.y)
        dribbled_distance = self.blackboard.dribbled_distance

        if robot.has_ball:
            if self.last_point_with_ball is not None:
                dribbled_distance += self.last_point_with_ball.distance_to(
                    current_point
                )
            self.last_point_with_ball = current_point
        else:
            self.last_point_with_ball = None
            dribbled_distance = 0.0
        
        self.blackboard.set(
            "dribbled_distance", dribbled_distance, overwrite=True
        )

        return py_trees.common.Status.SUCCESS


class DribbleMoveStep(AbstractBehaviour):
    """Moves the robot towards the target while dribbling the ball."""

    def __init__(self, dribble_limit: float, stop_dribble_ratio: float = 0.75):
        self.limit = dribble_limit
        self.stop_dribble_ratio = stop_dribble_ratio
        super().__init__()
    
    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="target_coords", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="dribbled_distance", access=py_trees.common.Access.READ
        )

    def update(self) -> py_trees.common.Status: 
        bb = self.blackboard     
        dribbled_dist = bb.dribbled_distance
        game = bb.game.current
        target = bb.target_coords
        robot = game.friendly_robots[bb.robot_id]
        
        print(f"Executing DribbleMoveStep for robot {self.blackboard.robot_id}, target{target}")
        
        dribble = True if dribbled_dist <= self.limit * self.stop_dribble_ratio else False
        current_point = Vector2D(robot.p.x, robot.p.y)
        if target is None:
            return py_trees.common.Status.FAILURE
        target_oren = current_point.angle_to(target)

        command = move(
            game=game,
            motion_controller=self.blackboard.motion_controller,
            robot_id=bb.robot_id,
            target_coords=target,
            target_oren=target_oren,
            dribbling=dribble,
        )
        self.blackboard.cmd_map[bb.robot_id] = command
        return py_trees.common.Status.RUNNING


class StopStep(AbstractBehaviour):
    """Issues an empty command to stop the robot and turn off the dribbler."""

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        command = empty_command()
        self.blackboard.cmd_map[self.blackboard.robot_id] = command
        return py_trees.common.Status.SUCCESS

