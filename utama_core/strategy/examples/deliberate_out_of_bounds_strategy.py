"""Opponent strategy that deliberately creates out-of-bounds ball-placement events."""

import math

import py_trees

from utama_core.config.field_params import FieldDimensions
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


def _angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


class DeliberateOutOfBoundsStep(AbstractBehaviour):
    """Line up behind the ball, shove once toward a boundary, then stop."""

    _DIRECTIONS = (
        Vector2D(0.0, 1.0),
        Vector2D(0.0, -1.0),
        Vector2D(-1.0, 0.0),
        Vector2D(1.0, 0.0),
    )
    _BOUNDARY_EPSILON = 0.04
    _APPROACH_BACKOFF = ROBOT_RADIUS + 0.10
    _APPROACH_TOLERANCE = 0.08
    _ALIGN_TOLERANCE_RAD = 0.14
    _PUSH_SPEED = 1.9
    _PUSH_SECONDS = 0.34
    _STOP_SECONDS = 0.35

    def __init__(self, field_dims: FieldDimensions, name: str = "DeliberateOutOfBoundsStep"):
        super().__init__(name=name)
        self._field_dims = field_dims
        self._direction_index = 0
        self._phase = "approach"
        self._phase_started_at = 0.0
        self._waiting_for_restart = False

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        if game.ball is None or not game.friendly_robots:
            return self._stop_all()

        ball_pos = Vector2D(game.ball.p.x, game.ball.p.y)
        half_length = self._field_dims.full_field_half_length
        half_width = self._field_dims.full_field_half_width

        if self._waiting_for_restart:
            if abs(ball_pos.x) < half_length - 0.25 and abs(ball_pos.y) < half_width - 0.25:
                self._waiting_for_restart = False
                self._phase = "approach"
            else:
                return self._stop_all()

        direction = self._DIRECTIONS[self._direction_index]
        if self._has_reached_target_boundary(ball_pos, direction, half_length, half_width):
            self._direction_index = (self._direction_index + 1) % len(self._DIRECTIONS)
            self._waiting_for_restart = True
            self._phase = "approach"
            return self._stop_all()

        pusher_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_pos),
        )
        pusher = game.friendly_robots[pusher_id]
        setup_point = Vector2D(
            ball_pos.x - direction.x * self._APPROACH_BACKOFF,
            ball_pos.y - direction.y * self._APPROACH_BACKOFF,
        )
        push_oren = math.atan2(direction.y, direction.x)

        for robot_id in game.friendly_robots:
            if robot_id != pusher_id:
                self.blackboard.cmd_map[robot_id] = empty_command(False)
                continue

            if self._phase == "approach":
                if pusher.p.distance_to(setup_point) <= self._APPROACH_TOLERANCE:
                    self._phase = "align"
                    self._phase_started_at = game.ts
                self.blackboard.cmd_map[robot_id] = move(
                    game,
                    motion_controller,
                    robot_id,
                    setup_point,
                    push_oren,
                    dribbling=False,
                )
                continue

            if self._phase == "align":
                aligned = abs(_angle_delta(push_oren, pusher.orientation)) <= self._ALIGN_TOLERANCE_RAD
                near_setup = pusher.p.distance_to(setup_point) <= self._APPROACH_TOLERANCE * 1.8
                if aligned and near_setup:
                    self._phase = "push"
                    self._phase_started_at = game.ts
                else:
                    self.blackboard.cmd_map[robot_id] = move(
                        game,
                        motion_controller,
                        robot_id,
                        setup_point,
                        push_oren,
                        dribbling=False,
                    )
                    continue

            if self._phase == "push":
                if game.ts - self._phase_started_at <= self._PUSH_SECONDS:
                    self.blackboard.cmd_map[robot_id] = RobotCommand(
                        local_forward_vel=self._PUSH_SPEED,
                        local_left_vel=0,
                        angular_vel=0,
                        kick=0,
                        chip=0,
                        dribble=0,
                    )
                    continue
                self._phase = "stop"
                self._phase_started_at = game.ts

            if self._phase == "stop":
                self.blackboard.cmd_map[robot_id] = empty_command(False)
                if game.ts - self._phase_started_at >= self._STOP_SECONDS:
                    self._phase = "approach"

        return py_trees.common.Status.RUNNING

    def _has_reached_target_boundary(
        self,
        ball_pos: Vector2D,
        direction: Vector2D,
        half_length: float,
        half_width: float,
    ) -> bool:
        if direction.y > 0.0:
            return ball_pos.y >= half_width - self._BOUNDARY_EPSILON
        if direction.y < 0.0:
            return ball_pos.y <= -half_width + self._BOUNDARY_EPSILON
        if direction.x < 0.0:
            return ball_pos.x <= -half_length + self._BOUNDARY_EPSILON
        return ball_pos.x >= half_length - self._BOUNDARY_EPSILON

    def _stop_all(self) -> py_trees.common.Status:
        game = self.blackboard.game
        for robot_id in game.friendly_robots:
            self.blackboard.cmd_map[robot_id] = empty_command(False)
        return py_trees.common.Status.RUNNING


class DeliberateOutOfBoundsStrategy(AbstractStrategy):
    """Opponent strategy that pushes the ball out: top, bottom, left, right."""

    def __init__(self, field_dims: FieldDimensions) -> None:
        self._field_dims = field_dims
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return n_runtime_friendly >= 1

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self):
        return None

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="DeliberateOutOfBoundsRoot", memory=False)
        root.add_child(DeliberateOutOfBoundsStep(self._field_dims))
        return root
