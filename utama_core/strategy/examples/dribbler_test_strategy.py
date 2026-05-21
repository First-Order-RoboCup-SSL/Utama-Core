"""Strategy that tests the dribbler: fetch ball, carry it forward, done."""

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

_CARRY_DIST = 0.5  # metres to carry the ball forward
_FETCH_TOL = 0.15  # metres — close enough to start carrying
_ARRIVE_TOL = 0.10  # metres — done


class DribblerStep(AbstractBehaviour):
    """
    Fetch ball without dribbler, carry it forward with dribbler on, then stop.

    States:
        FETCH  — approach ball, dribbler off
        CARRY  — dribbler on, drive forward _CARRY_DIST metres from fetch position
        DONE   — dribbler off, idle
    """

    def __init__(self, robot_id_key: str):
        super().__init__()
        self.robot_id_key = robot_id_key
        self._state = "FETCH"
        self._target: Vector2D | None = None

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        robot_id: int = self.blackboard.get(self.robot_id_key)

        robot = game.friendly_robots[robot_id]
        ball = game.ball.p.to_2d()

        if self._state == "FETCH":
            cmd = go_to_ball(game, self.blackboard.motion_controller, robot_id, dribble_when_near=False)
            dist_to_ball = math.dist((robot.p.x, robot.p.y), (ball.x, ball.y))
            if dist_to_ball < _FETCH_TOL:
                # Target is _CARRY_DIST forward (toward enemy goal) from current ball position
                goal_x = game.field.enemy_goal_line[0][0]
                forward = 1.0 if goal_x > 0 else -1.0
                self._target = Vector2D(ball.x + forward * _CARRY_DIST, ball.y)
                self._state = "CARRY"

        elif self._state == "CARRY":
            target = self._target
            if rsim_env:
                rsim_env.draw_point(target.x, target.y, color="blue")
            face_oren = robot.p.angle_to(target)
            cmd = move(game, self.blackboard.motion_controller, robot_id, target, face_oren, dribbling=True)
            if math.dist((robot.p.x, robot.p.y), (target.x, target.y)) < _ARRIVE_TOL:
                self._state = "DONE"

        else:  # DONE
            cmd = empty_command(dribbler_on=False)

        self.blackboard.cmd_map[robot_id] = cmd
        return py_trees.common.Status.RUNNING


class DribblerTestStrategy(AbstractStrategy):
    """1-robot strategy: fetch ball → carry forward with dribbler → stop."""

    def __init__(self, robot_id: int = 0):
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return n_runtime_friendly == 1 and n_runtime_enemy == 0

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> SpaceRequirements:
        return SpaceRequirements(min_length=1.0, min_width=0.5)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        robot_id_key = "target_robot_id"

        root = Sequence(name="DribblerRoot", memory=False)
        root.add_children(
            [
                SetBlackboardVariable("SetRobotID", robot_id_key, self.robot_id),
                DribblerStep(robot_id_key=robot_id_key),
            ]
        )
        return root
