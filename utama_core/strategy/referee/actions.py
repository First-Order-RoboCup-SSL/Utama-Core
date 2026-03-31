"""Referee action nodes for each referee game state.

Each node:
  - Reads game state from blackboard.game
  - Writes robot commands to blackboard.cmd_map for every friendly robot
  - Returns RUNNING (the parent Selector holds here until the command changes)

All positions are in the ssl-vision coordinate system (metres).
Team side is resolved at tick-time via game.my_team_is_yellow and
game.my_team_is_right so no construction-time team colour is needed.
"""

import math

import py_trees

from utama_core.config.referee_constants import (
    BALL_KEEP_OUT_DISTANCE,
    BALL_PLACEMENT_DONE_DISTANCE,
    CLEARANCE_FALLBACK_DIRECTION,
    KICKOFF_DEFENCE_POSITION_RATIOS_RIGHT,
    KICKOFF_SUPPORT_POSITION_RATIOS_RIGHT,
    PENALTY_BEHIND_MARK_DISTANCE,
    PENALTY_LINE_Y_STEP_RATIO,
    PENALTY_MARK_HALF_FIELD_RATIO,
)
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import Field
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour


def _all_stop(blackboard) -> py_trees.common.Status:
    """Send empty_command to every friendly robot and return RUNNING."""
    for robot_id in blackboard.game.friendly_robots:
        blackboard.cmd_map[robot_id] = empty_command(False)
    return py_trees.common.Status.RUNNING


def _field_half_length(game) -> float:
    """Return the current field half-length, supporting Field and FieldBounds alike."""
    field = game.field
    if hasattr(field, "half_length"):
        return field.half_length
    return (field.bottom_right[0] - field.top_left[0]) / 2.0


def _field_half_width(game) -> float:
    """Return the current field half-width, supporting Field and FieldBounds alike."""
    field = game.field
    if hasattr(field, "half_width"):
        return field.half_width
    return (field.top_left[1] - field.bottom_right[1]) / 2.0


def _penalty_mark_x(goal_x: float) -> float:
    """Place the penalty mark midway between centre and goal line."""
    return goal_x * PENALTY_MARK_HALF_FIELD_RATIO


def _scaled_position(game, x_ratio: float, y_ratio: float) -> Vector2D:
    """Scale a normalized formation coordinate to the current field dimensions."""
    return Vector2D(x_ratio * _field_half_length(game), y_ratio * _field_half_width(game))


def _ensure_outside_center_circle(target: Vector2D) -> Vector2D:
    """Project kickoff support points to the centre-circle boundary when needed."""
    dist = math.hypot(target.x, target.y)
    keep_radius = Field.CENTER_CIRCLE_RADIUS
    if dist == 0.0 or dist >= keep_radius:
        return target
    scale = keep_radius / dist
    return Vector2D(target.x * scale, target.y * scale)


def _formation_positions(game, ratios: tuple[tuple[float, float], ...], mirror_x: bool) -> list[Vector2D]:
    """Build field-scaled formation positions, mirroring x when we defend the left goal."""
    positions = []
    for x_ratio, y_ratio in ratios:
        scaled_x_ratio = -x_ratio if mirror_x else x_ratio
        positions.append(_ensure_outside_center_circle(_scaled_position(game, scaled_x_ratio, y_ratio)))
    return positions


def _clear_from_ball(blackboard, keep_dist: float = BALL_KEEP_OUT_DISTANCE) -> py_trees.common.Status:
    """Move encroaching robots out beyond the keep-out radius and stop the rest."""
    game = blackboard.game
    ball = game.ball
    motion_controller = blackboard.motion_controller
    if ball is None:
        return _all_stop(blackboard)

    bx, by = ball.p.x, ball.p.y
    for robot_id, robot in game.friendly_robots.items():
        dx = robot.p.x - bx
        dy = robot.p.y - by
        dist = math.hypot(dx, dy)
        if dist >= keep_dist:
            blackboard.cmd_map[robot_id] = empty_command(False)
            continue

        if dist == 0.0:
            ux, uy = CLEARANCE_FALLBACK_DIRECTION
        else:
            ux, uy = dx / dist, dy / dist

        target = Vector2D(bx + ux * keep_dist, by + uy * keep_dist)
        oren = robot.p.angle_to(target)
        blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, target, oren)

    return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# HALT — zero velocity, highest priority
# ---------------------------------------------------------------------------


class HaltStep(AbstractBehaviour):
    """Sends zero-velocity commands to all friendly robots.

    Required: robots must stop immediately on HALT (2-second grace period allowed).
    """

    def update(self) -> py_trees.common.Status:
        return _all_stop(self.blackboard)


# ---------------------------------------------------------------------------
# STOP — stop in place (≤1.5 m/s; ≥0.5 m from ball)
# Stopping cold satisfies both constraints.
# ---------------------------------------------------------------------------


class StopStep(AbstractBehaviour):
    """Moves encroaching robots out of the keep-out radius and stops the rest."""

    def update(self) -> py_trees.common.Status:
        return _clear_from_ball(self.blackboard)


# ---------------------------------------------------------------------------
# BALL PLACEMENT — ours
# ---------------------------------------------------------------------------


class BallPlacementOursStep(AbstractBehaviour):
    """Moves the closest friendly robot to place the ball at designated_position.

    If the chosen placer does not yet have the ball, it first drives to the ball
    with the dribbler on. Once it has possession, it carries the ball to the
    designated position. All other robots stop in place.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        # Determine which team is ours
        our_team = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        if getattr(our_team, "can_place_ball", None) is False:
            return _all_stop(self.blackboard)

        target = ref.designated_position
        if target is None:
            return _all_stop(self.blackboard)

        target_pos = Vector2D(target[0], target[1])
        ball = game.ball
        if ball is None:
            return _all_stop(self.blackboard)

        if ball.p.distance_to(target_pos) <= BALL_PLACEMENT_DONE_DISTANCE:
            return _all_stop(self.blackboard)

        # Pick the placer: robot closest to the ball
        placer_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(ball.p),
        )

        for robot_id in game.friendly_robots:
            if robot_id == placer_id:
                robot = game.friendly_robots[robot_id]
                if robot.has_ball:
                    target_for_move = target_pos
                else:
                    target_for_move = ball.p
                oren = robot.p.angle_to(target_for_move)
                self.blackboard.cmd_map[robot_id] = move(
                    game, motion_controller, robot_id, target_for_move, oren, dribbling=True
                )
            else:
                self.blackboard.cmd_map[robot_id] = empty_command(False)

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# BALL PLACEMENT — theirs
# ---------------------------------------------------------------------------


class BallPlacementTheirsStep(AbstractBehaviour):
    """Actively clear our robots away from the ball during their placement."""

    def update(self) -> py_trees.common.Status:
        return _clear_from_ball(self.blackboard)


# ---------------------------------------------------------------------------
# PREPARE_KICKOFF — ours
# ---------------------------------------------------------------------------


class PrepareKickoffOursStep(AbstractBehaviour):
    """Positions robots for our kickoff.

    Robot with the lowest ID approaches the ball at (0, 0).
    All other robots move to own-half support positions outside the centre circle.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        robot_ids = sorted(game.friendly_robots.keys())
        kicker_id = robot_ids[0]

        # Support positions depend on which side we defend
        support_positions = _formation_positions(
            game,
            KICKOFF_SUPPORT_POSITION_RATIOS_RIGHT,
            mirror_x=not game.my_team_is_right,
        )

        support_idx = 0
        for robot_id in robot_ids:
            if robot_id == kicker_id:
                # Approach the ball at centre, face the opponent goal
                target = Vector2D(0.0, 0.0)
                goal_x = _field_half_length(game) if not game.my_team_is_right else -_field_half_length(game)
                oren = math.atan2(0.0 - target.y, goal_x - target.x)
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, target, oren)
            else:
                pos = support_positions[support_idx % len(support_positions)]
                support_idx += 1
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, pos, 0.0)

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# PREPARE_KICKOFF — theirs
# ---------------------------------------------------------------------------


class PrepareKickoffTheirsStep(AbstractBehaviour):
    """Moves all our robots to own half, outside the centre circle, for the opponent kickoff."""

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        positions = _formation_positions(
            game,
            KICKOFF_DEFENCE_POSITION_RATIOS_RIGHT,
            mirror_x=not game.my_team_is_right,
        )

        for idx, robot_id in enumerate(sorted(game.friendly_robots.keys())):
            pos = positions[idx % len(positions)]
            self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, pos, 0.0)

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# PREPARE_PENALTY — ours
# ---------------------------------------------------------------------------


class PreparePenaltyOursStep(AbstractBehaviour):
    """Positions robots for our penalty kick.

    Kicker (lowest non-keeper ID): moves to our penalty mark, faces goal.
    All others: stop on a line behind the penalty mark on our side.

    Penalty mark is placed halfway between the centre line and the target goal line.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        # Our goalkeeper ID from the referee packet
        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        # Opponent goal is on the right if we are on the right, else on the left
        field_half_length = _field_half_length(game)
        opp_goal_x = field_half_length if not game.my_team_is_right else -field_half_length
        sign = 1 if not game.my_team_is_right else -1
        penalty_mark = Vector2D(_penalty_mark_x(opp_goal_x), 0.0)
        behind_line_x = penalty_mark.x - sign * PENALTY_BEHIND_MARK_DISTANCE

        goal_oren = math.atan2(0.0, opp_goal_x - penalty_mark.x)

        robot_ids = sorted(game.friendly_robots.keys())
        non_keeper_ids = [rid for rid in robot_ids if rid != keeper_id]
        kicker_id = non_keeper_ids[0] if non_keeper_ids else robot_ids[0]

        behind_idx = 0
        behind_y_step = PENALTY_LINE_Y_STEP_RATIO * _field_half_width(game)
        for robot_id in robot_ids:
            if robot_id == kicker_id:
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, penalty_mark, goal_oren)
            else:
                # Place behind the line, spread in y
                offset = (behind_idx - (len(robot_ids) - 1) / 2.0) * behind_y_step
                pos = Vector2D(behind_line_x, offset)
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, pos, 0.0)
                behind_idx += 1

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# PREPARE_PENALTY — theirs
# ---------------------------------------------------------------------------


class PreparePenaltyTheirsStep(AbstractBehaviour):
    """Positions our robots for the opponent's penalty kick.

    Goalkeeper: moves to our goal line centre.
    All others: move to a line behind the penalty mark on our half.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        # Our goal is on the right if my_team_is_right, else on the left
        field_half_length = _field_half_length(game)
        our_goal_x = field_half_length if game.my_team_is_right else -field_half_length
        sign = 1 if game.my_team_is_right else -1

        # Opponent's penalty mark is in our half, between centre and our goal line.
        opp_penalty_mark_x = _penalty_mark_x(our_goal_x)
        behind_line_x = opp_penalty_mark_x + sign * PENALTY_BEHIND_MARK_DISTANCE

        robot_ids = sorted(game.friendly_robots.keys())
        behind_idx = 0
        behind_y_step = PENALTY_LINE_Y_STEP_RATIO * _field_half_width(game)

        for robot_id in robot_ids:
            if robot_id == keeper_id:
                # Keeper on own goal line, facing the incoming ball
                keeper_pos = Vector2D(our_goal_x, 0.0)
                self.blackboard.cmd_map[robot_id] = move(
                    game, motion_controller, robot_id, keeper_pos, math.pi if game.my_team_is_right else 0.0
                )
            else:
                offset = (behind_idx - (len(robot_ids) - 1) / 2.0) * behind_y_step
                pos = Vector2D(behind_line_x, offset)
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, pos, 0.0)
                behind_idx += 1

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# DIRECT_FREE — ours
# ---------------------------------------------------------------------------


class DirectFreeOursStep(AbstractBehaviour):
    """Positions our robots for our direct free kick.

    The robot closest to the ball becomes the kicker and drives toward the ball.
    All other robots stop in place (they may be repositioned by the strategy tree
    after NORMAL_START transitions the override layer to pass-through).
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller
        ball = game.ball

        kicker_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(ball.p) if ball else float("inf"),
        )

        for robot_id in game.friendly_robots:
            if robot_id == kicker_id and ball:
                robot = game.friendly_robots[robot_id]
                oren = robot.p.angle_to(ball.p)
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, ball.p, oren)
            else:
                self.blackboard.cmd_map[robot_id] = empty_command(False)

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# DIRECT_FREE — theirs
# ---------------------------------------------------------------------------


class DirectFreeTheirsStep(AbstractBehaviour):
    """Actively clear our robots out of the ball keep-out radius."""

    def update(self) -> py_trees.common.Status:
        return _clear_from_ball(self.blackboard)


# ---------------------------------------------------------------------------
# Helper: resolve bilateral commands
# ---------------------------------------------------------------------------


def is_our_command(command: RefereeCommand, our_command: RefereeCommand, their_command: RefereeCommand) -> bool:
    """Not used directly — bilateral resolution is done in tree.py via command sets."""
    return command == our_command
