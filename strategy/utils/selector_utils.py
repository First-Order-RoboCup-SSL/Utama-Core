import py_trees
from strategy.common.abstract_behaviour import AbstractBehaviour

class HasBall(AbstractBehaviour):
    """
    Checks if the specified robot currently has possession of the ball.

    This behavior is a condition that reads the `has_ball` attribute of a
    robot from the game state. It's used to verify if a robot has
    successfully collected the ball.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot to check for ball possession. Typically from the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the robot has the ball.
        - `py_trees.common.Status.FAILURE`: Otherwise.
    """

    def __init__(self, name="HasBall", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self):
        super().setup()
        
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self):
        # print(f"Checking if robot {self.blackboard.robot_id} has the ball")
        if self.blackboard.game.current.friendly_robots[
            self.blackboard.robot_id
        ].has_ball:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class GoalScored(AbstractBehaviour):
    """
    Checks if a goal has been scored by either team.

    This behavior is a condition that checks if the ball has crossed either
    side of the pitch by examining its x-coordinate. It provides a simple
    way to detect a score and trigger a change in strategy.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the ball's absolute x-coordinate > 4.5.
        - `py_trees.common.Status.FAILURE`: Otherwise.
    """

    def __init__(self, name="GoalScored", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)
        
    def update(self):
        if abs(self.blackboard.game.current.ball.p.x) > 4.5:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
