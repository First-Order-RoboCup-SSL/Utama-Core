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

    _TARGETS = ("top", "bottom", "left", "right", "yellow_goal")
    _BOUNDARY_EPSILON = 0.04
    _GOAL_EPSILON = 0.06
    _GOAL_MOUTH_CLEARANCE = 0.28
    _APPROACH_BACKOFF = ROBOT_RADIUS + 0.10
    _APPROACH_TOLERANCE = 0.08
    _ALIGN_TOLERANCE_RAD = 0.14
    _PUSH_SPEED = 1.9
    _PUSH_SECONDS = 0.34
    _STOP_SECONDS = 0.35

    def __init__(self, field_dims: FieldDimensions, name: str = "DeliberateOutOfBoundsStep"):
        super().__init__(name=name)
        self._field_dims = field_dims
        self._target_index = 0
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

        target_name = self._TARGETS[self._target_index]
        if self._has_reached_target(ball_pos, target_name, game, half_length, half_width):
            if self._target_index < len(self._TARGETS) - 1:
                self._target_index += 1
                self._waiting_for_restart = True
            self._phase = "approach"
            return self._stop_all()

        target_point = self._target_point(target_name, ball_pos, game, half_length, half_width)
        direction = target_point - ball_pos
        if direction.mag() == 0.0:
            return self._stop_all()
        direction = Vector2D(direction.x / direction.mag(), direction.y / direction.mag())

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

    def _target_point(
        self,
        target_name: str,
        ball_pos: Vector2D,
        game,
        half_length: float,
        half_width: float,
    ) -> Vector2D:
        if target_name == "top":
            return Vector2D(ball_pos.x, half_width + self._BOUNDARY_EPSILON)
        if target_name == "bottom":
            return Vector2D(ball_pos.x, -half_width - self._BOUNDARY_EPSILON)
        if target_name == "left":
            return Vector2D(-half_length - self._BOUNDARY_EPSILON, self._safe_touchline_y(ball_pos.y, game))
        if target_name == "right":
            return Vector2D(half_length + self._BOUNDARY_EPSILON, self._safe_touchline_y(ball_pos.y, game))

        yellow_is_right = game.my_team_is_right == game.my_team_is_yellow
        yellow_goal_x = half_length + self._GOAL_EPSILON if yellow_is_right else -half_length - self._GOAL_EPSILON
        return Vector2D(yellow_goal_x, 0.0)

    def _safe_touchline_y(self, current_y: float, game) -> float:
        min_abs_y = game.field.half_goal_width + self._GOAL_MOUTH_CLEARANCE
        sign = 1.0 if current_y >= 0.0 else -1.0
        return sign * max(abs(current_y), min_abs_y)

    def _has_reached_target(
        self,
        ball_pos: Vector2D,
        target_name: str,
        game,
        half_length: float,
        half_width: float,
    ) -> bool:
        if target_name == "top":
            return ball_pos.y >= half_width - self._BOUNDARY_EPSILON
        if target_name == "bottom":
            return ball_pos.y <= -half_width + self._BOUNDARY_EPSILON
        if target_name == "left":
            return ball_pos.x <= -half_length + self._BOUNDARY_EPSILON and abs(ball_pos.y) > game.field.half_goal_width
        if target_name == "right":
            return ball_pos.x >= half_length - self._BOUNDARY_EPSILON and abs(ball_pos.y) > game.field.half_goal_width

        yellow_is_right = game.my_team_is_right == game.my_team_is_yellow
        if yellow_is_right:
            return ball_pos.x >= half_length - self._GOAL_EPSILON and abs(ball_pos.y) < game.field.half_goal_width
        return ball_pos.x <= -half_length + self._GOAL_EPSILON and abs(ball_pos.y) < game.field.half_goal_width

    def _stop_all(self) -> py_trees.common.Status:
        game = self.blackboard.game
        for robot_id in game.friendly_robots:
            self.blackboard.cmd_map[robot_id] = empty_command(False)
        return py_trees.common.Status.RUNNING


class DeliberateOutOfBoundsStrategy(AbstractStrategy):
    """Opponent strategy that pushes the ball out once per side, then scores on yellow."""

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
