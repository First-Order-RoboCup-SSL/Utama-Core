from typing import Any, Optional

import py_trees
from py_trees.composites import Selector, Sequence

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


class HasBall(AbstractBehaviour):
    """
    Checks whether the configured robot currently owns the ball.
    **Args:**
        - visual (bool): If True, uses distance-based possession check; otherwise uses game state.
    **Blackboard Interaction:**
        - Reads:
            - `robot_id` (int): The ID of the robot to check for ball possession.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The robot has possession of the ball.
        - `py_trees.common.Status.FAILURE`: The robot does not have possession of the ball.
    """

    def __init__(self, rd_robot_id: str, visual: bool = True, name: Optional[str] = None):
        super().__init__(name)
        self.robot_id_key = rd_robot_id

        self.visual = visual
        DISTANCE_BUFFER = 0.04  # meters
        self.ball_capture_dist = DISTANCE_BUFFER + ROBOT_RADIUS

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def update(self):
        # print(f"Checking if robot {self.blackboard.get(self.robot_id_key)} has the ball")
        game = self.blackboard.game
        robot_id = self.blackboard.get(self.robot_id_key)

        if self.visual:
            return self._has_ball_visual(game, robot_id)
        else:
            return self._has_ball_from_state(game, robot_id)

    def _has_ball_visual(self, game: Game, robot_id: int) -> py_trees.common.Status:
        """
        Visual possession: success if the robot is within `ball_capture_radius` of the ball.
        Uses squared distance (no sqrt) for speed.
        """
        robot = game.friendly_robots[robot_id]
        ball = game.ball

        r_pos = Vector2D(robot.p.x, robot.p.y)
        b_pos = Vector2D(ball.p.x, ball.p.y)

        dist_sq = r_pos.distance_to(b_pos)
        return py_trees.common.Status.SUCCESS if dist_sq < self.ball_capture_dist else py_trees.common.Status.FAILURE

    def _has_ball_from_state(self, game: Game, robot_id: int) -> py_trees.common.Status:
        """
        State possession: success if the game state's `has_ball` flag is true.
        """
        has_ball = game.current.friendly_robots[robot_id].has_ball

        # print(f"Robot {robot_id} has_ball: {has_ball}")
        return py_trees.common.Status.SUCCESS if has_ball else py_trees.common.Status.FAILURE


class GoToBallStep(AbstractBehaviour):
    """
    Continuously moves the robot towards the ball using the core go_to_ball skill.
    **Blackboard Interaction:**
        - Reads:
            - `robot_id` (int): The ID of the robot to move towards the ball.
    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to go to the ball.
    """

    def __init__(self, rd_robot_id: str, name: str = "GoToBallStep"):
        super().__init__(name)
        self.robot_id_key = rd_robot_id

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        env = self.blackboard.rsim_env
        robot_id = self.blackboard.get(self.robot_id_key)
        if env:
            v = game.friendly_robots[robot_id].v
            p = game.friendly_robots[robot_id].p
            env.draw_point(p.x + v.x * 0.0167 * 5, p.y + v.y * 0.0167 * 5, color="green")

        command = go_to_ball(game, self.blackboard.motion_controller, robot_id)
        self.blackboard.cmd_map[robot_id] = command
        return py_trees.common.Status.RUNNING


class SetBlackboardVariable(AbstractBehaviour):
    """
    Writes a constant `value` onto the blackboard with the key `variable_name`.
    **Blackboard Interaction:**
        - Writes:
            - `variable_name` (Any): The name of the blackboard variable to be set.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The variable has been set.
    """

    def __init__(self, name: str, variable_name: str, value: Any):
        super().__init__(name=name)
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS


def go_to_ball_subtree(rd_robot_id: str) -> py_trees.behaviour.Behaviour:
    """
    Builds a selector that drives the robot to the ball until it gains possession.
    **Args:**
        - rd_robot_id: Blackboard key that holds the robot ID to control. **Required.**
    **Status:**
        - py_trees.common.Status.SUCCESS: The robot already has the ball.
        - py_trees.common.Status.RUNNING: The robot is being commanded to go to the ball.
    **Returns:**
        - py_trees.behaviour.Behaviour: The root node of the Go-To-Ball subtree.
    """
    root = Selector(
        name="GoToBallSubtree",
        memory=False,
    )

    ### Assemble the tree ###

    root.add_children(
        [
            HasBall(name="HasBall?", rd_robot_id=rd_robot_id),
            GoToBallStep(name="GoToBallStep", rd_robot_id=rd_robot_id),
        ]
    )

    return root


class GoToBallExampleStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the GoToBallStrategy with a specific robot ID.
        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        self.robot_id_key = "robot_id"
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self):
        return None

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete go_to_ball behaviour tree."""

        root = Sequence(name="GoToBallRoot", memory=True)

        set_robot_id = SetBlackboardVariable(
            name="SetRobotID",
            variable_name=self.robot_id_key,
            value=self.robot_id,
        )

        ### Assemble the tree ###

        root.add_children([set_robot_id, go_to_ball_subtree(rd_robot_id=self.robot_id_key)])

        return root
