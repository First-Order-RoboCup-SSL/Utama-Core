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
    Checks if the robot is oriented towards a target with a dynamic threshold.

    This behavior is a condition that succeeds if the robot's current
    orientation is sufficiently close to the target orientation. The acceptable
    error margin decreases as the robot gets closer to the goal, ensuring
    higher precision is required before taking a shot.

    **Blackboard Interaction:
        Reads:
            - `robot_id` (int): The ID of the robot to check.
            - `target_orientation` (float): The desired orientation angle in radians. typically from the `ShouldScoreGoal` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the orientation is within the threshold.
        - `py_trees.common.Status.FAILURE`: Otherwise.
    """

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="best_shot", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="target_orientation", access=py_trees.common.Access.READ
        )

    def initialise(self):
        self.goal_x = self.blackboard.game.field.enemy_goal_line.coords[0][0]

    def update(self):
        shooter = self.blackboard.game.current.friendly_robots[self.blackboard.robot_id]
        shot_orientation = self.blackboard.get("target_orientation")

        threshold = abs(shooter.orientation - shot_orientation) * abs(
            self.goal_x - shooter.p.x
        )

        if threshold <= 0.02:
            # print(f"Robot {self.blackboard.robot_id} is oriented correctly towards the goal.")
            return py_trees.common.Status.SUCCESS
        else:
            # print(f"Robot {self.blackboard.robot_id} is NOT oriented correctly towards the goal threshold {threshold}. {shooter.orientation} vs {shot_orientation}")
            return py_trees.common.Status.FAILURE


class GoalBlocked(AbstractBehaviour):
    """
    Checks if the calculated best shot path to the goal is blocked by an opponent.

    This behavior is a condition that simulates a line from the ball to the
    best shot position on the goal line. It then checks if any enemy robot
    is intersecting this path.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the shooting robot (for context). Set through the `SetBlackboardVariable` node.
            - `best_shot` (float): The y-coordinate of the optimal shot target on the goal line. typically from the `ShouldScoreGoal` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the path is blocked by an opponent.
        - `py_trees.common.Status.FAILURE`: If the path is clear.
    """

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="best_shot", access=py_trees.common.Access.READ
        )

    def initialise(self):
        self.goal_x = self.blackboard.game.field.enemy_goal_line.coords[0][0]

    def update(self):
        self.enemy_robots = list(self.blackboard.game.enemy_robots.values())
        self.ball = self.blackboard.game.ball
        self.best_shot = self.blackboard.get("best_shot")

        if self._is_goal_blocked():
            # print(f"Goal is blocked by enemy robots. Best shot position: {self.best_shot}")
            return py_trees.common.Status.SUCCESS
        else:
            # print(f"Goal is not blocked. Best shot position: {self.best_shot}")
            return py_trees.common.Status.FAILURE

    def _is_goal_blocked(self) -> bool:
        """
        Determines whether the goal is blocked by enemy robots (considering them as circles).
        Determines whether the goal is blocked by enemy robots (considering them as circles).

        :param game: The game state containing robot and ball positions.
        :return: True if the goal is blocked, False otherwise.
        """

        ball_x, ball_y = self.ball.p.x, self.ball.p.y

        # Define the shooting line from ball position in the shooter's direction
        line_start = np.array([ball_x, ball_y])
        # print(f"Best shot position: {self.best_shot}")
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
    Evaluates if a viable shot on goal exists and calculates its quality.

    This behavior analyzes the field to find the best possible shot by identifying
    the largest open angle to the enemy goal line, considering opponent positions.
    It calculates a "shot quality" metric based on the size of the open angle
    and the robot's distance to the goal.

    If a viable shot is found, it writes the optimal target y-coordinate and
    the target orientation to the blackboard for other behaviors to use.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the potential shooting robot. Set with the `SetBlackboardVariable` node.
        Writes:
            - `best_shot` (float): The optimal y-coordinate on the goal line for the shot.
            - `target_orientation` (float): The orientation (in radians) required to make the shot.

    **Returns:**
        - py_trees.common.Status.SUCCESS: If the calculated shot quality is > 0. **(Temporarily set to 0 for testing)**
        - py_trees.common.Status.FAILURE: If no valid shot is found or the quality is 0.
    """

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="best_shot", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="target_orientation", access=py_trees.common.Access.WRITE
        )

    def initialise(self):
        self.target_goal_line = self.blackboard.game.field.enemy_goal_line

        self.goal_x = self.target_goal_line.coords[0][0]
        self.goal_y1 = self.target_goal_line.coords[1][1]
        self.goal_y2 = self.target_goal_line.coords[0][1]

    def update(self):
        self.shooter = self.blackboard.game.friendly_robots[self.blackboard.robot_id]
        self.enemy_robots = self.blackboard.game.enemy_robots
        self.ball = self.blackboard.game.ball

        shot_quality = self._find_shot_quality()
        if shot_quality > 0.3:
            print(f"Shot quality SUCCESS: {shot_quality}")
            return py_trees.common.Status.SUCCESS
        else:
            # print(f"Shot quality FAILURE: {shot_quality}")
            return py_trees.common.Status.FAILURE

    def _find_shot_quality(
        self,
    ) -> float:
        """
        Computes the shot quality based on the open angle to the goal / total angle to the goal.
        Uses the _find_best_shot function to determine the largest open angle.
        """

        # Full angle between the two goalposts
        full_angle = angle_between_points(
            self.shooter.p,
            Vector2D(self.goal_x, self.goal_y1),
            Vector2D(self.goal_x, self.goal_y2),
        )

        # Use _find_best_shot to get the largest gap
        best_shot, largest_gap = _find_best_shot(
            self.shooter.p,
            list(self.enemy_robots.values()),
            self.goal_x,
            self.goal_y1,
            self.goal_y2,
        )

        if best_shot is not None:
            self.blackboard.set("best_shot", best_shot, overwrite=True)
            shot_orientation = np.arctan2(
                (best_shot - self.shooter.p.y), (self.goal_x - self.shooter.p.x)
            )
            self.blackboard.set("target_orientation", shot_orientation, overwrite=True)
        else:
            print("No valid shot found.")
            return 0.0

        # Compute the open angle (gap angle)
        open_angle = angle_between_points(
            self.shooter.p,
            Vector2D(self.goal_x, largest_gap[0]),
            Vector2D(self.goal_x, largest_gap[1]),
        )

        distance_to_goal_ratio = (
            np.absolute(self.shooter.p.x - self.goal_x)
        ) / np.absolute(2 * self.goal_x)

        distance_to_goal_weight = 0.4

        # Normalize shot quality
        shot_quality = (
            open_angle / full_angle - distance_to_goal_weight * distance_to_goal_ratio
            if full_angle > 0
            else 0
        )
        return shot_quality
