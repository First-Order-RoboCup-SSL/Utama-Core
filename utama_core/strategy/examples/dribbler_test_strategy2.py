"""Strategy that tests the dribbler through a fixed sequence: forward → left → right → back → stop."""

import math

import py_trees
from py_trees.composites import Sequence

from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import (
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.strategy.examples.utils import SetBlackboardVariable

_CARRY_DIST = 0.4  # metres per leg
_ARRIVE_TOL = 0.10


class DribblerSequenceStep(AbstractBehaviour):
    """
    Fetch ball until has_ball, then carry through a fixed sequence of legs, dribbler off when done.

    Sequence after fetch:
        1. Forward (toward enemy goal)
        2. Left  (+y)
        3. Right (-y, back across)
        4. Backward (away from enemy goal)
        5. DONE — dribbler off, idle
    """

    def __init__(self, robot_id_key: str):
        super().__init__()
        self.robot_id_key = robot_id_key
        self._state = "FETCH"
        self._legs: list[Vector2D] = []
        self._leg_idx = 0
        self._origin: Vector2D | None = None

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def _build_legs(self, fetch_pos: Vector2D, forward: float) -> list[Vector2D]:
        """Absolute waypoints for each carry leg, computed once at fetch time."""
        x0, y0 = fetch_pos.x, fetch_pos.y
        p1 = Vector2D(x0 + forward * _CARRY_DIST, y0)  # forward
        p2 = Vector2D(x0 + forward * _CARRY_DIST, y0 + _CARRY_DIST)  # left
        p3 = Vector2D(x0 + forward * _CARRY_DIST, y0 - _CARRY_DIST)  # right
        p4 = Vector2D(x0, y0 - _CARRY_DIST)  # back
        return [p1, p2, p3, p4]

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        robot_id: int = self.blackboard.get(self.robot_id_key)

        robot = game.friendly_robots[robot_id]
        ball = game.ball.p.to_2d()

        if self._state == "FETCH":
            cmd = go_to_ball(game, self.blackboard.motion_controller, robot_id, dribble_when_near=False)
            if robot.has_ball:
                goal_x = game.field.enemy_goal_line[0][0]
                forward = 1.0 if goal_x > 0 else -1.0
                self._legs = self._build_legs(ball, forward)
                self._leg_idx = 0
                self._state = "CARRY"

        elif self._state == "CARRY":
            target = self._legs[self._leg_idx]
            if rsim_env:
                rsim_env.draw_point(target.x, target.y, color="blue")

            face_oren = robot.p.angle_to(target)
            cmd = move(game, self.blackboard.motion_controller, robot_id, target, face_oren, dribbling=True)

            if math.dist((robot.p.x, robot.p.y), (target.x, target.y)) < _ARRIVE_TOL:
                self._leg_idx += 1
                if self._leg_idx >= len(self._legs):
                    self._state = "DONE"

        else:  # DONE
            cmd = empty_command(dribbler_on=False)

        self.blackboard.cmd_map[robot_id] = cmd
        return py_trees.common.Status.RUNNING


class DribblerSequenceStrategy(AbstractStrategy):
    """1-robot strategy: fetch → carry forward/left/right/back → stop."""

    def __init__(self, robot_id: int = 0):
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return n_runtime_friendly == 1 and n_runtime_enemy == 0

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> SpaceRequirements:
        return SpaceRequirements(min_length=1.0, min_width=1.0)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        robot_id_key = "target_robot_id"

        root = Sequence(name="DribblerSeqRoot", memory=False)
        root.add_children(
            [
                SetBlackboardVariable("SetRobotID", robot_id_key, self.robot_id),
                DribblerSequenceStep(robot_id_key=robot_id_key),
            ]
        )
        return root
