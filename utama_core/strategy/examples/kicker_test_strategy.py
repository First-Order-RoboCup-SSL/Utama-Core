"""Strategy that tests the kicker: go to ball, align to goal, kick."""

import math

import py_trees
from py_trees.composites import Sequence

from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import empty_command, kick, turn_on_spot
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import (
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.strategy.examples.utils import SetBlackboardVariable

_ALIGN_TOL = 0.06  # radians
_KICK_TICKS = 3
_WAIT_TICKS = 40


class KickerStep(AbstractBehaviour):
    """
    Go to ball, turn to face goal, kick, wait, repeat.

    States:
        FETCH — go_to_ball until has_ball
        ALIGN — turn on spot to face enemy goal
        KICK  — issue kick for _KICK_TICKS ticks
        WAIT  — idle while ball travels
    """

    def __init__(self, robot_id_key: str):
        super().__init__()
        self.robot_id_key = robot_id_key
        self._state = "FETCH"
        self._kick_ticks = 0
        self._wait_ticks = 0

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        robot_id: int = self.blackboard.get(self.robot_id_key)

        robot = game.friendly_robots[robot_id]

        goal_x = game.field.enemy_goal_line[0][0]
        goal = Vector2D(goal_x, 0.0)
        ball = game.ball.p.to_2d()
        shoot_oren = ball.angle_to(goal)

        if rsim_env:
            rsim_env.draw_point(goal.x, goal.y, color="red")

        if self._state == "FETCH":
            cmd = go_to_ball(game, self.blackboard.motion_controller, robot_id, dribble_when_near=True)
            if robot.has_ball:
                self._state = "ALIGN"

        elif self._state == "ALIGN":
            cmd = turn_on_spot(game, self.blackboard.motion_controller, robot_id, shoot_oren, dribbling=True)
            oren_err = abs((robot.orientation - shoot_oren + math.pi) % (2 * math.pi) - math.pi)
            if oren_err < _ALIGN_TOL:
                self._kick_ticks = 0
                self._state = "KICK"

        elif self._state == "KICK":
            cmd = kick()
            self._kick_ticks += 1
            if self._kick_ticks >= _KICK_TICKS:
                self._wait_ticks = 0
                self._state = "WAIT"

        else:  # WAIT
            cmd = empty_command()
            self._wait_ticks += 1
            if self._wait_ticks > _WAIT_TICKS:
                self._state = "FETCH"

        self.blackboard.cmd_map[robot_id] = cmd
        return py_trees.common.Status.RUNNING


class KickerTestStrategy(AbstractStrategy):
    """1-robot strategy: fetch ball → align to goal → kick → repeat."""

    def __init__(self, robot_id: int = 0):
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return n_runtime_friendly == 1 and n_runtime_enemy == 0

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> SpaceRequirements:
        return SpaceRequirements(min_length=2.0, min_width=1.0)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        robot_id_key = "target_robot_id"

        root = Sequence(name="KickerRoot", memory=False)
        root.add_children(
            [
                SetBlackboardVariable("SetRobotID", robot_id_key, self.robot_id),
                KickerStep(robot_id_key=robot_id_key),
            ]
        )
        return root
