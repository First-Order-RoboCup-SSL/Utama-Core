import py_trees
from skills.src.score_goal import _find_best_shot
from entities.data.vector import Vector2D
from strategy.common.abstract_behaviour import AbstractBehaviour
from global_utils.math_utils import angle_between_points
from entities.game import Game, Robot, Ball
from typing import List, Tuple

from config.settings import ROBOT_RADIUS

import numpy as np

class OrenAtTargetThreshold(AbstractBehaviour):
    """
    A condition behaviour that checks if the robot's orientation is within a threshold of the target orientation.
    Requires `robot_id` to be set in the blackboard prior.
    """

    def __init__(self, name="OrientationAtTarget", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="best_shot", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_orientation", access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.goal_x = self.blackboard.game.field.enemy_goal_line.coords[0][0]
    
    def update(self):
        ball = self.blackboard.game.current.ball
        shooter = self.blackboard.game.current.friendly_robots[self.blackboard.robot_id]
        shot_orientation = np.atan2((self.blackboard.best_shot - ball.p.y), (self.goal_x - ball.p.x)) % (2 * np.pi)
        
        self.blackboard.set("target_orientation", shot_orientation, overwrite=True)
        
        threshold = abs(shooter.orientation - shot_orientation) * abs(
            self.goal_x - shooter.p.x
        )

        if threshold <= 0.02:
            print(f"Robot {self.blackboard.robot_id} is oriented correctly towards the goal.")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"Robot {self.blackboard.robot_id} is NOT oriented correctly towards the goal threshold {threshold}. {shooter.orientation} vs {shot_orientation}")
            return py_trees.common.Status.FAILURE

class GoalBlocked(AbstractBehaviour):
    """
    A condition behaviour that checks if the robot has the ball.
    Requires `robot_id` to be set in the blackboard prior.
    """

    def __init__(self, name="GoalBlocked", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="best_shot", access=py_trees.common.Access.READ)
    
    def initialise(self):
        self.goal_x = self.blackboard.game.field.enemy_goal_line.coords[0][0]
    
    def update(self):
        self.enemy_robots = list(self.blackboard.game.enemy_robots.values())
        self.ball = self.blackboard.game.ball
        self.best_shot = self.blackboard.get("best_shot")
        
        if self.is_goal_blocked():
            print(f"Goal is blocked by enemy robots. Best shot position: {self.best_shot}")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"Goal is not blocked. Best shot position: {self.best_shot}")
            return py_trees.common.Status.FAILURE

    def is_goal_blocked(self) -> bool:
        """
        Determines whether the goal is blocked by enemy robots (considering them as circles).
        Determines whether the goal is blocked by enemy robots (considering them as circles).

        :param game: The game state containing robot and ball positions.
        :return: True if the goal is blocked, False otherwise.
        """

        ball_x, ball_y = self.ball.p.x, self.ball.p.y

        # Define the shooting line from ball position in the shooter's direction
        line_start = np.array([ball_x, ball_y])
        print(f"Best shot position: {self.best_shot}")
        line_end = np.array([self.goal_x, self.best_shot])  # Use the best shot position

        # Helper function: shortest distance from a point to a line segment
        def distance_point_to_line(point, line_start, line_end):
            line_vec = line_end - line_start
            point_vec = point - line_start
            proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
            proj = np.clip(proj, 0, 1)  # Keep projection within the segment
            closest_point = line_start + proj * line_vec
            return np.linalg.norm(point - closest_point)

        # Check if any enemy robot blocks the shooting path
        for defender in self.enemy_robots:
            if defender:
                robot_pos = np.array([defender.p.x, defender.p.y])
                distance = distance_point_to_line(robot_pos, line_start, line_end)

                if distance <= ROBOT_RADIUS:  # Consider robot as a circle
                    return True  # Shot is blocked

        return False  # No robot is blocking

class ShouldScoreGoal(AbstractBehaviour):
    """
    A condition behaviour that checks if a goal has been scored.
    Requires `robot_id` to be set in the blackboard prior.
    """

    def __init__(self, name="ShouldScoreGoal", opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="best_shot", access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.target_goal_line = self.blackboard.game.field.enemy_goal_line
        
        self.goal_x = self.target_goal_line.coords[0][0]
        self.goal_y1 = self.target_goal_line.coords[1][1]
        self.goal_y2 = self.target_goal_line.coords[0][1]

    def update(self):
        self.shooter = self.blackboard.game.friendly_robots[self.blackboard.robot_id]
        self.enemy_robots = self.blackboard.game.enemy_robots
        self.ball = self.blackboard.game.ball
        
        shot_quality = self.find_shot_quality()
        if shot_quality:
            print(f"Shot quality SUCCESS: {shot_quality}")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"Shot quality FAILURE: {shot_quality}")
            return py_trees.common.Status.FAILURE
    
    def find_shot_quality(
        self,
    ) -> float:
        """
        Computes the shot quality based on the open angle to the goal / total angle to the goal.
        Uses the _find_best_shot function to determine the largest open angle.
        """

        # Full angle between the two goalposts
        full_angle = angle_between_points(
            self.shooter.p, Vector2D(self.goal_x, self.goal_y1), Vector2D(self.goal_x, self.goal_y2)
        )

        # Use _find_best_shot to get the largest gap
        best_shot, largest_gap = _find_best_shot(
            self.shooter.p, list(self.enemy_robots.values()), self.goal_x, self.goal_y1, self.goal_y2
        )

        self.blackboard.set(
            "best_shot", best_shot, overwrite=True
        )
        
        # Compute the open angle (gap angle)
        open_angle = angle_between_points(
            self.shooter.p,
            Vector2D(self.goal_x, largest_gap[0]),
            Vector2D(self.goal_x, largest_gap[1]),
        )

        distance_to_goal_ratio = (np.absolute(self.shooter.p.x - self.goal_x)) / np.absolute(2 * self.goal_x)

        distance_to_goal_weight = 0.4

        # Normalize shot quality
        shot_quality = (
            open_angle / full_angle - distance_to_goal_weight * distance_to_goal_ratio
            if full_angle > 0
            else 0
        )
        return shot_quality
      
