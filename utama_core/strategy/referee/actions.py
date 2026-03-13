"""Hardcoded action nodes for each referee game state.

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

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour

# SSL Div B field constants (metres)
_PENALTY_MARK_DIST = 6.0  # distance from goal centre to penalty mark
_HALF_FIELD_X = 4.5  # half field length
_CENTRE_CIRCLE_R = 0.5  # centre circle radius
_BALL_KEEP_DIST = 0.55  # ≥0.5 m required; 5 cm buffer
_PENALTY_BEHIND_OFFSET = 0.4  # robots must be ≥0.4 m behind penalty mark
_OPP_DEF_AREA_KEEP_DIST = 0.25  # ≥0.2 m from opponent defence area; 5 cm buffer


def _all_stop(blackboard) -> py_trees.common.Status:
    """Send empty_command to every friendly robot and return RUNNING."""
    for robot_id in blackboard.game.friendly_robots:
        blackboard.cmd_map[robot_id] = empty_command(False)
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
    """Sends zero-velocity commands to all friendly robots.

    Complies with STOP: robots are stationary, so speed = 0 m/s ≤ 1.5 m/s
    and they do not approach the ball.
    """

    def update(self) -> py_trees.common.Status:
        return _all_stop(self.blackboard)


# ---------------------------------------------------------------------------
# BALL PLACEMENT — ours
# ---------------------------------------------------------------------------


class BallPlacementOursStep(AbstractBehaviour):
    """Moves the closest friendly robot to the designated_position to place the ball.

    All other robots stop in place. If can_place_ball is False, all robots stop.

    The placing robot drives toward designated_position using the move() skill.
    Ball capture and release are handled by the dribbler (future: dribble_subtree).
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        # Determine which team is ours
        our_team = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        if not getattr(our_team, "can_place_ball", True):
            return _all_stop(self.blackboard)

        target = ref.designated_position
        if target is None:
            return _all_stop(self.blackboard)

        target_pos = Vector2D(target[0], target[1])
        ball = game.ball

        # Pick the placer: robot closest to the ball
        placer_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(ball.p) if ball else float("inf"),
        )

        for robot_id in game.friendly_robots:
            if robot_id == placer_id:
                # Face the target while approaching
                robot = game.friendly_robots[robot_id]
                oren = robot.p.angle_to(target_pos)
                self.blackboard.cmd_map[robot_id] = move(
                    game, motion_controller, robot_id, target_pos, oren, dribbling=True
                )
            else:
                self.blackboard.cmd_map[robot_id] = empty_command(False)

        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# BALL PLACEMENT — theirs
# ---------------------------------------------------------------------------


class BallPlacementTheirsStep(AbstractBehaviour):
    """Stops all friendly robots during the opponent's ball placement.

    Robots stopped in place are guaranteed not to approach the ball or interfere
    with the placement. Active clearance (move ≥0.5 m from ball) is a future
    enhancement.
    """

    def update(self) -> py_trees.common.Status:
        return _all_stop(self.blackboard)


# ---------------------------------------------------------------------------
# PREPARE_KICKOFF — ours
# ---------------------------------------------------------------------------

# Kickoff formation positions (own half, outside centre circle).
# Relative x is negative = own half when we are on the right; sign is flipped below.
_KICKOFF_SUPPORT_POSITIONS_RIGHT = [
    Vector2D(-0.8, 0.5),
    Vector2D(-0.8, -0.5),
    Vector2D(-1.5, 0.8),
    Vector2D(-1.5, -0.8),
    Vector2D(-2.5, 0.0),
]
_KICKOFF_SUPPORT_POSITIONS_LEFT = [Vector2D(-p.x, p.y) for p in _KICKOFF_SUPPORT_POSITIONS_RIGHT]


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
        support_positions = (
            _KICKOFF_SUPPORT_POSITIONS_RIGHT if game.my_team_is_right else _KICKOFF_SUPPORT_POSITIONS_LEFT
        )

        support_idx = 0
        for robot_id in robot_ids:
            if robot_id == kicker_id:
                # Approach the ball at centre, face the opponent goal
                target = Vector2D(0.0, 0.0)
                goal_x = _HALF_FIELD_X if not game.my_team_is_right else -_HALF_FIELD_X
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

_KICKOFF_DEFENCE_POSITIONS_RIGHT = [
    Vector2D(-0.8, 0.4),
    Vector2D(-0.8, -0.4),
    Vector2D(-1.5, 0.6),
    Vector2D(-1.5, -0.6),
    Vector2D(-2.5, 0.0),
    Vector2D(-1.5, 0.0),
]
_KICKOFF_DEFENCE_POSITIONS_LEFT = [Vector2D(-p.x, p.y) for p in _KICKOFF_DEFENCE_POSITIONS_RIGHT]


class PrepareKickoffTheirsStep(AbstractBehaviour):
    """Moves all our robots to own half, outside the centre circle, for the opponent kickoff."""

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        positions = _KICKOFF_DEFENCE_POSITIONS_RIGHT if game.my_team_is_right else _KICKOFF_DEFENCE_POSITIONS_LEFT

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
    All others: stop on a line 0.4 m behind the penalty mark (on own side).

    Penalty mark is at (opp_goal_x ∓ 6.0, 0), sign depends on which side we attack.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        # Our goalkeeper ID from the referee packet
        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        # Opponent goal is on the right if we are on the right, else on the left
        opp_goal_x = _HALF_FIELD_X if not game.my_team_is_right else -_HALF_FIELD_X
        sign = 1 if not game.my_team_is_right else -1
        penalty_mark = Vector2D(opp_goal_x - sign * _PENALTY_MARK_DIST, 0.0)
        behind_line_x = penalty_mark.x - sign * _PENALTY_BEHIND_OFFSET

        goal_oren = math.atan2(0.0, opp_goal_x - penalty_mark.x)

        robot_ids = sorted(game.friendly_robots.keys())
        non_keeper_ids = [rid for rid in robot_ids if rid != keeper_id]
        kicker_id = non_keeper_ids[0] if non_keeper_ids else robot_ids[0]

        behind_idx = 0
        behind_y_step = 0.35
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
    All others: move to a line 0.4 m behind the penalty mark on our half.
    """

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        # Our goal is on the right if my_team_is_right, else on the left
        our_goal_x = _HALF_FIELD_X if game.my_team_is_right else -_HALF_FIELD_X
        sign = 1 if game.my_team_is_right else -1

        # Opponent's penalty mark is in their half attacking our goal
        opp_penalty_mark_x = our_goal_x - sign * _PENALTY_MARK_DIST
        behind_line_x = opp_penalty_mark_x + sign * _PENALTY_BEHIND_OFFSET

        robot_ids = sorted(game.friendly_robots.keys())
        behind_idx = 0
        behind_y_step = 0.35

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
    """Stops all our robots during the opponent's direct free kick.

    All robots must remain ≥ 0.5 m from the ball. Stopping in place satisfies this
    assuming robots are not already within 0.5 m (future: add active clearance).
    """

    def update(self) -> py_trees.common.Status:
        return _all_stop(self.blackboard)


# ---------------------------------------------------------------------------
# Helper: resolve bilateral commands
# ---------------------------------------------------------------------------


def is_our_command(command: RefereeCommand, our_command: RefereeCommand, their_command: RefereeCommand) -> bool:
    """Not used directly — bilateral resolution is done in tree.py via command sets."""
    return command == our_command
