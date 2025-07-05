import numpy as np
import py_trees
from strategy.common.abstract_behaviour import AbstractBehaviour

from typing import Dict

class HasBall(AbstractBehaviour):
    """
    A condition behaviour that checks if the robot has the ball.
    Requires `robot_id` to be set in the blackboard prior.
    """

    def __init__(self, name="HasBall", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        
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
    A condition behaviour that checks if a goal has been scored.
    Requires `robot_id` to be set in the blackboard prior.
    """

    def __init__(self, name="GoalScored", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self):
        if abs(self.blackboard.game.current.ball.p.x) > 4.5:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
