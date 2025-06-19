from typing import Any
import py_trees

from robot_control.src.skills import go_to_ball
from strategy.behaviour_trees.behaviours.robocup_behaviour import RobocupBehaviour

class SetBlackboardVariable(RobocupBehaviour):
    """A generic behaviour to set a variable on the blackboard."""
    def __init__(self, name: str, variable_name: str, value: Any):
        super().__init__(name=name)
        self.variable_name = variable_name
        self.value = value
        self.blackboard.register_key(
            key=self.variable_name, access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS

class GoToBallStep(RobocupBehaviour):
    """A behaviour that executes a single step of the go_to_ball skill."""
    def __init__(self, name="GoToBallStep"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        command =       (
            self.blackboard.present_future_game.current,
            self.blackboard.pid_oren,
            self.blackboard.pid_trans,
            self.blackboard.robot_id,
        )
        self.blackboard.robot_controller.add_robot_commands(
            command, self.blackboard.robot_id
        )
        return py_trees.common.Status.RUNNING

class HasBall(RobocupBehaviour):
    """A condition behaviour that checks if the robot has the ball."""
    def __init__(self, name="HasBall"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.present_future_game.current.friendly_robots[self.blackboard.robot_id].has_ball:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

# Factory function to create a complete go_to_ball behaviour tree.
def create_go_to_ball_behaviour(robot_id: int) -> py_trees.behaviour.Behaviour:
    """Factory function to create a complete go_to_ball behaviour tree."""
    
    # Root sequence for the whole behaviour
    root = py_trees.composites.Sequence(name="GoToBall", memory=True)
    
    # A child sequence to set the robot_id on the blackboard
    set_robot_id = SetBlackboardVariable(
        name="SetTargetRobotID", variable_name="robot_id", value=robot_id
    )

    # A selector to decide whether to get the ball or stop
    has_ball_selector = py_trees.composites.Selector(name="HasBallSelector", memory=False)
    has_ball_selector.add_child(HasBall())
    has_ball_selector.add_child(GoToBallStep())

    # Assemble the tree
    root.add_child(set_robot_id)
    root.add_child(has_ball_selector)
    
    return root