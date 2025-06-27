import py_trees
from strategy.common.abstract_behaviour import AbstractBehaviour


class SetWhoHasBall(AbstractBehaviour):
    """
    A behaviour that sets the robot ID of the robot that has the ball on the blackboard.
    This is used to determine which robot should take actions related to the ball.

    If a robot has the ball, it will set the `HasBallRobotID` key in the blackboard and return success.
    Else if no robot has the ball, it will set `HasBallRobotID` to -1 and return failure.
    """

    def __init__(self, name="SetWhoHasBall"):
        super().__init__(name=name)
        self.blackboard.register_key(
            key="HasBallRobotID", access=py_trees.common.Access.WRITE
        )

    def update(self):
        for (
            robot_id,
            robot,
        ) in self.blackboard.present_future_game.current.friendly_robots.items():
            if robot.has_ball:
                self.blackboard.set(name="HasBallRobotID", value=robot_id)
                return py_trees.common.Status.SUCCESS
        self.blackboard.set(name="HasBallRobotID", value=-1)
        return py_trees.common.Status.FAILURE


class HasBall(AbstractBehaviour):
    """
    A condition behaviour that checks if the robot has the ball.
    Requires `robot_id` to be set in the blackboard.
    """

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
