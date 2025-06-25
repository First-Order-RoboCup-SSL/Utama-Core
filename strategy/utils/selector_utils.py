import py_trees
from strategy.common.abstract_behaviour import AbstractBehaviour


class HasBall(AbstractBehaviour):
    """A condition behaviour that checks if the robot has the ball."""

    def __init__(self, name="HasBall"):
        super().__init__(name=name)
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.present_future_game.current.friendly_robots[
            self.blackboard.robot_id
        ].has_ball:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
