"""Referee action nodes for each referee game state.

Each node:
  - Reads game state from blackboard.game
  - Writes robot commands to blackboard.cmd_map for every friendly robot

All positions are in the ssl-vision coordinate system (metres).
Team side is resolved at tick-time via game.my_team_is_yellow and
game.my_team_is_right so no construction-time team colour is needed.
"""

import math

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.config.referee_constants import (
    BALL_KEEP_OUT_DISTANCE,
    BALL_PLACEMENT_DONE_DISTANCE,
    KICKOFF_DEFENCE_POSITION_RATIOS_OWN_HALF,
    KICKOFF_SUPPORT_POSITION_RATIOS_OWN_HALF,
    OPPONENT_DEFENSE_AREA_KEEP_DISTANCE,
    PENALTY_BEHIND_MARK_DISTANCE,
    PENALTY_LINE_Y_STEP_RATIO,
    PENALTY_MARK_HALF_FIELD_RATIO,
)
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.tolerance import Sticky
from utama_core.skills.src.utils.move_utils import empty_command, move, turn_on_spot


def _all_stop(blackboard) -> None:
    """Send empty_command to every friendly robot."""
    for robot_id in blackboard.game.friendly_robots:
        blackboard.cmd_map[robot_id] = empty_command(False)


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
    """Project kickoff support points outside the ball keep-out zone if needed.

    Uses BALL_KEEP_OUT_DISTANCE (not just the centre-circle radius) so that
    formation positions are never closer than our enforced clearance distance,
    avoiding any need for _clear_to_legal_positions to further adjust them and
    preventing robots from being placed on a path that crosses the keep-out zone.
    """
    dist = math.hypot(target.x, target.y)
    if dist == 0.0 or dist >= BALL_KEEP_OUT_DISTANCE:
        return target
    scale = BALL_KEEP_OUT_DISTANCE / dist
    return Vector2D(target.x * scale, target.y * scale)


def _formation_positions(game, ratios: tuple[tuple[float, float], ...]) -> list[Vector2D]:
    """Build own-half field-scaled formation positions for the current defended side."""
    positions = []
    own_half_sign = 1.0 if game.my_team_is_right else -1.0
    for x_ratio, y_ratio in ratios:
        positions.append(_ensure_outside_center_circle(_scaled_position(game, own_half_sign * x_ratio, y_ratio)))
    return positions


def _project_outside_circle(
    point: Vector2D,
    center: Vector2D,
    keep_dist: float,
    fallback_direction: tuple[float, float],
) -> Vector2D:
    """Project a point to the circle boundary if it lies inside the keep-out radius."""
    offset = point - center
    dist = offset.mag()
    if dist >= keep_dist:
        return point
    if dist == 0.0:
        ux, uy = fallback_direction
        return Vector2D(center.x + ux * keep_dist, center.y + uy * keep_dist)
    scale = keep_dist / dist
    return Vector2D(center.x + offset.x * scale, center.y + offset.y * scale)


def _clamp_to_field(point: Vector2D, game) -> Vector2D:
    """Clamp a position to within the field boundaries with a small inset margin."""
    margin = 0.1
    half_length = _field_half_length(game) - margin
    half_width = _field_half_width(game) - margin
    return Vector2D(
        max(-half_length, min(half_length, point.x)),
        max(-half_width, min(half_width, point.y)),
    )


def _clamp_to_field_or_ball(point: Vector2D, game, ball_pos: Vector2D) -> Vector2D:
    """Like `_clamp_to_field`, but never clamps a point *farther* from the ball
    than it already is.

    A direct-free/ball-placement restart is routinely awarded exactly when the
    ball has gone out of bounds, so an approach point derived from the ball's
    real position (a fixed offset toward the field, e.g. `DirectFreeOursStep`'s
    kick-approach point) can itself sit just outside the line — clamping that
    straight to `_clamp_to_field`'s inset margin then strands the robot ~0.1m
    inside the line while the ball sits farther out, permanently short of
    `_KICKER_READY_DIST`/placement-done range with no way to close the gap
    (confirmed live: a full-length tournament match froze in DIRECT_FREE_YELLOW
    for the rest of the game this way — the kicker converged to exactly
    ball_pos.x clamped to -half_length+0.1, ~0.31m from a ball resting ~0.21m
    past the line). The field boundary isn't a physical wall in SSL, so a
    robot briefly crossing it to reach a ball that's legitimately out there is
    fine; `_clamp_to_field` only exists to keep *other* geometry (formation
    spots, clearing targets) from drifting to absurd off-field points, not to
    block a ball-approach target from reaching the ball itself.
    """
    clamped = _clamp_to_field(point, game)
    if (clamped - ball_pos).mag() <= (point - ball_pos).mag():
        return clamped
    return point


def _is_in_own_defense_area(game, x: float, y: float) -> bool:
    """Return True if (x, y) is inside our own defense area."""
    own_goal_sign = 1.0 if game.my_team_is_right else -1.0
    field_half_length = _field_half_length(game)
    depth = game.field.half_defense_area_depth
    width = game.field.half_defense_area_width
    # Own defense area inner edge (closest to center)
    inner_x = own_goal_sign * (field_half_length - 2.0 * depth)
    if own_goal_sign > 0:
        return x >= inner_x and abs(y) <= width
    else:
        return x <= inner_x and abs(y) <= width


def _own_defense_area_exit_x(game) -> float:
    """Return the x-coordinate just outside our own defense area front edge."""
    own_goal_sign = 1.0 if game.my_team_is_right else -1.0
    field_half_length = _field_half_length(game)
    depth = game.field.half_defense_area_depth
    inner_x = own_goal_sign * (field_half_length - 2.0 * depth)
    # Step one robot-diameter outside the front edge
    return inner_x - own_goal_sign * (2 * ROBOT_RADIUS + 0.05)


def _project_outside_opp_defense_area(game, point: Vector2D, keep_dist: float) -> Vector2D:
    """Project a point out of the opponent defense area plus the required keep distance."""
    field_half_length = _field_half_length(game)
    opp_goal_sign = -1.0 if game.my_team_is_right else 1.0
    defense_width = game.field.half_defense_area_width + keep_dist
    defense_inner_x = opp_goal_sign * (field_half_length - 2.0 * game.field.half_defense_area_depth)
    safe_x = defense_inner_x - opp_goal_sign * keep_dist

    if abs(point.y) > defense_width:
        return point

    if opp_goal_sign < 0.0:
        if point.x >= safe_x:
            return point
    else:
        if point.x <= safe_x:
            return point

    return Vector2D(safe_x, point.y)


# A detour waypoint sits on a circle this much wider than the keep-out one, at least
# `_DETOUR_MIN_TURN_RAD` further round from the robot. The straight leg to it from a robot
# on the keep-out edge stays outside while 1.1 * cos(turn) >= 1, i.e. turn <= 24.6deg.
_DETOUR_RADIUS_FACTOR = 1.1
_DETOUR_MIN_TURN_RAD = math.radians(20.0)


def _detour_around_circle(start: Vector2D, target: Vector2D, center: Vector2D, radius: float) -> Vector2D:
    """`target`, or a waypoint round the circle if the straight line from `start` to
    `target` passes through it: the tangent point (or `_DETOUR_MIN_TURN_RAD` further
    round, once the robot is on the circle), on the side the target lies. Called every
    tick, so the robot heads for the real target as soon as that line is clear.

    Found on tournament_20260924_124033: after a goal the conceding team is still in
    the scorer's half (STOP only clears robots near the ball). At PREPARE_KICKOFF its
    robots drove straight at their own-half spots through the centre circle, and
    `keep_out` voided 47 of 54 kickoffs after goals into a FORCE_START scramble."""
    seg = target - start
    length_sq = seg.dot(seg)
    if length_sq == 0.0:
        return target
    t = min(1.0, max(0.0, (center - start).dot(seg) / length_sq))
    if (start + seg * t - center).mag() >= radius:
        return target
    rel_start, rel_target = start - center, target - center
    dist = rel_start.mag()
    if dist == 0.0:
        return target  # on the centre itself: the caller's push-out handles it
    wide = radius * _DETOUR_RADIUS_FACTOR
    turn = max(math.acos(min(1.0, wide / dist)), _DETOUR_MIN_TURN_RAD)
    cross = rel_start.x * rel_target.y - rel_start.y * rel_target.x
    angle = math.atan2(rel_start.y, rel_start.x) + (turn if cross >= 0.0 else -turn)
    return Vector2D(center.x + wide * math.cos(angle), center.y + wide * math.sin(angle))


def _clear_to_legal_positions(
    blackboard,
    *,
    ball_keep_dist: float | None = None,
    designated_keep_dist: float | None = None,
    clear_opp_defense_area: bool = False,
    clear_own_defense_area: bool = False,
    max_own_defenders: int = 1,
    exempt_robot_ids: set[int] | None = None,
    intended_targets: dict[int, Vector2D] | None = None,
) -> None:
    """Move encroaching robots to the nearest legal location and stop the rest.

    If intended_targets is provided, each robot's clearance starts from its
    intended formation target rather than its current position, so the clearing
    pass refines rather than discards the formation intent.

    If clear_own_defense_area is True, robots in excess of max_own_defenders
    inside our own defense area are pushed out to just outside the front edge.
    The goalkeeper (lowest-ID robot inside the area) is kept as the allowed defender.
    """
    game = blackboard.game
    motion_controller = blackboard.motion_controller
    exempt_robot_ids = exempt_robot_ids or set()

    # Determine which friendly robots must exit own defense area.
    own_defense_evict: set[int] = set()
    if clear_own_defense_area:
        in_own_area = [
            rid
            for rid, robot in game.friendly_robots.items()
            if rid not in exempt_robot_ids and _is_in_own_defense_area(game, robot.p.x, robot.p.y)
        ]
        # Keep the first max_own_defenders (sorted by id) — evict the rest.
        for rid in sorted(in_own_area)[max_own_defenders:]:
            own_defense_evict.add(rid)

    ball_center = None
    if ball_keep_dist is not None and game.ball is not None:
        ball_center = Vector2D(game.ball.p.x, game.ball.p.y)

    designated_center = None
    ref = game.referee
    if designated_keep_dist is not None and ref is not None and ref.designated_position is not None:
        designated_center = Vector2D(ref.designated_position[0], ref.designated_position[1])

    # When a robot is exactly coincident with an obstruction center (dist == 0),
    # push it toward own half — robots should be on their side in all restart states.
    own_half_sign = 1.0 if game.my_team_is_right else -1.0
    own_half_fallback = (own_half_sign, 0.0)
    ball_fallback = own_half_fallback
    designated_fallback = own_half_fallback

    for robot_id, robot in game.friendly_robots.items():
        if robot_id in exempt_robot_ids:
            continue

        robot_pos = Vector2D(robot.p.x, robot.p.y)

        # If the robot is currently inside any keep-out zone, clear it from its
        # current position immediately — don't head toward a distant formation
        # target that requires traversing the exclusion zone first.
        currently_encroaching = (ball_center is not None and (ball_center - robot_pos).mag() < ball_keep_dist) or (
            designated_center is not None and (designated_center - robot_pos).mag() < designated_keep_dist
        )
        intended = intended_targets.get(robot_id) if intended_targets is not None else None

        if not currently_encroaching and intended_targets is not None and robot_id in intended_targets:
            target = intended_targets[robot_id]
        else:
            target = robot_pos

        if robot_id in own_defense_evict:
            target = Vector2D(_own_defense_area_exit_x(game), robot.p.y)
        if ball_center is not None:
            target = _project_outside_circle(target, ball_center, ball_keep_dist, ball_fallback)
        if designated_center is not None:
            target = _project_outside_circle(target, designated_center, designated_keep_dist, designated_fallback)
        if clear_opp_defense_area:
            target = _project_outside_opp_defense_area(game, target, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)

        if ball_center is not None and not currently_encroaching:
            target = _detour_around_circle(robot_pos, target, ball_center, ball_keep_dist)
        elif ball_center is not None and intended is not None:
            # Out to the edge and on round toward the formation spot in one go. Pushed out
            # alone, a robot stops a hair inside the edge, stays "encroaching", and its target
            # stays its own position: robots parked on the circle in the wrong half at kickoff.
            spot = _project_outside_circle(intended, ball_center, ball_keep_dist, ball_fallback)
            target = _detour_around_circle(target, spot, ball_center, ball_keep_dist)

        target = _clamp_to_field(target, game)

        if target == Vector2D(robot.p.x, robot.p.y):
            blackboard.cmd_map[robot_id] = empty_command(False)
            continue

        oren = robot.p.angle_to(target)
        blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, target, oren)


# ---------------------------------------------------------------------------
# HALT — zero velocity, highest priority
# ---------------------------------------------------------------------------


class HaltStep:
    """Sends zero-velocity commands to all friendly robots.

    Required: robots must stop immediately on HALT (2-second grace period allowed).
    """

    def update(self) -> None:
        _all_stop(self.blackboard)


# ---------------------------------------------------------------------------
# STOP — stop in place (≤1.5 m/s; ≥0.5 m from ball)
# Stopping cold satisfies both constraints.
# ---------------------------------------------------------------------------


class StopStep:
    """Moves encroaching robots out of the keep-out radius and stops the rest."""

    def update(self) -> None:
        _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            clear_opp_defense_area=True,
            clear_own_defense_area=True,
            max_own_defenders=1,
        )


# ---------------------------------------------------------------------------
# BALL PLACEMENT — ours
# ---------------------------------------------------------------------------


class BallPlacementOursStep:
    """Moves the closest friendly robot to place the ball at designated_position.

    If the chosen placer does not yet have the ball, it first drives to the ball
    with the dribbler on. Once it has possession, it carries the ball to the
    designated position. All other robots clear away from the ball.
    """

    _RELEASE_DELAY_SECONDS = 0.25

    # How far *behind* the ball (opposite the carry direction) the robot
    # center's target sits, so it's the dribbler -- forward of center, not
    # the chassis origin -- that ends up on the ball. Unlike
    # DirectFreeOursStep._APPROACH_OFFSET (ROBOT_RADIUS + 0.03, a stand-off
    # distance that's fine for kicking since a kick only needs proximity/
    # aim), placement needs genuine possession: rsim's real has_ball contact
    # sensor only latches within ~0.081m forward of center (sslconfig.h's
    # distanceCenterKicker; see pass_and_score_geometry.has_ball's
    # docstring), so the target must bring the center essentially onto that
    # contact point, not stop a full robot-radius short of it. Confirmed
    # live, 2026-09-12: with the ROBOT_RADIUS+0.03 offset the placer parked
    # at rel_fwd=0.120 (just outside real-sensor range) and has_ball never
    # latched, freezing placement the same way driving straight to ball_pos
    # did (rel_fwd/rel_lat both 0.000, chassis overshooting onto the ball).
    _APPROACH_OFFSET = 0.10

    # Ball speed below which placement is considered settled enough to
    # start releasing. Without this, "close enough" (BALL_PLACEMENT_DONE_
    # DISTANCE alone) fires while the placer is still mid-approach at full
    # carry speed -- a robot covering the last stretch at ~1.4 m/s needs
    # ~0.5m to brake (v^2/2*a_max), well past the 0.15m done radius, so the
    # dribbler released there while the ball still has that speed and it
    # coasts on unguided. Confirmed live, 2026-09-12: ball measured moving
    # at 1.4 m/s when it first crossed the done radius, released, and
    # coasted from 0.150m to over 0.25m past target with the placer already
    # parked and no one re-chasing it -- froze BALL_PLACEMENT_YELLOW for the
    # rest of the match. Comfortably below DirectFreeOursStep's collision-
    # leniency COLLISION_SPEED_THRESHOLD_MPS (1.5 m/s, tuned for a different
    # purpose -- deciding whether a *moving obstacle* is safe to plan
    # through, not whether a placement has actually come to rest).
    _SETTLED_SPEED_MPS = 0.3

    # Same value/rationale as DirectFreeOursStep._KICKER_REASSIGN_MARGIN_M
    # and pass_and_shoot.py's _REASSIGN_MARGIN_M: metres a challenger must be
    # closer by before the placer role actually flips.
    _PLACER_REASSIGN_MARGIN_M = 0.3

    def __init__(self):
        self._release_started_at: float | None = None
        self._placer_id: int | None = None
        self._placer_sticky = Sticky[int](margin=self._PLACER_REASSIGN_MARGIN_M)

    def _reset_release(self) -> None:
        self._release_started_at = None
        self._placer_id = None
        self._placer_sticky.current = None

    def update(self) -> None:
        game = self.blackboard.game
        ref = game.referee
        motion_controller = self.blackboard.motion_controller

        # Determine which team is ours
        our_team = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        if getattr(our_team, "can_place_ball", None) is False:
            self._reset_release()
            _all_stop(self.blackboard)
            return

        target = ref.designated_position
        if target is None:
            self._reset_release()
            _all_stop(self.blackboard)
            return

        target_pos = Vector2D(target[0], target[1])
        ball = game.ball
        if ball is None:
            self._reset_release()
            _all_stop(self.blackboard)
            return

        ball_settled = math.hypot(ball.v.x, ball.v.y) <= self._SETTLED_SPEED_MPS
        if ball.p.distance_to(target_pos) <= BALL_PLACEMENT_DONE_DISTANCE and ball_settled:
            if self._release_started_at is None:
                self._release_started_at = game.ts

            placer_id = self._placer_id
            if placer_id not in game.friendly_robots:
                placer_id = min(
                    game.friendly_robots,
                    key=lambda rid: game.friendly_robots[rid].p.distance_to(ball.p),
                )

            hold_dribbler = game.ts - self._release_started_at < self._RELEASE_DELAY_SECONDS
            for robot_id in game.friendly_robots:
                self.blackboard.cmd_map[robot_id] = empty_command(hold_dribbler and robot_id == placer_id)
            return

        self._release_started_at = None

        # Pick the placer: robot closest to the ball, sticky across ticks
        # (see _PLACER_REASSIGN_MARGIN_M) -- same bug shape as
        # DirectFreeOursStep's kicker-identity thrashing (roadmap item 15):
        # a bare min(..., key=distance) recomputed fresh every tick flips
        # identity on ordinary sim noise whenever two-plus robots are
        # near-tied, resetting whoever newly "wins" to a standing start.
        # Narrower window here than DirectFreeOursStep's (non-placer robots
        # are actively cleared away every tick via _clear_to_legal_positions
        # below, so a tie self-resolves within a tick or two rather than
        # persisting for the whole restart) -- applied as hardening against
        # the same known-bad pattern, not confirmed as the root cause of any
        # specific stall.
        distances = {rid: game.friendly_robots[rid].p.distance_to(ball.p) for rid in game.friendly_robots}
        placer_id = self._placer_sticky.update(list(distances), score_fn=lambda rid: -distances[rid])
        self._placer_id = placer_id

        _FACE_READY_ANGLE = 0.2

        for robot_id in game.friendly_robots:
            if robot_id == placer_id:
                robot = game.friendly_robots[robot_id]
                if robot.has_ball:
                    oren = robot.p.angle_to(target_pos)
                    face_error = math.atan2(math.sin(oren - robot.orientation), math.cos(oren - robot.orientation))
                    if abs(face_error) > _FACE_READY_ANGLE:
                        self.blackboard.cmd_map[robot_id] = turn_on_spot(
                            game, motion_controller, robot_id, oren, dribbling=True
                        )
                    else:
                        self.blackboard.cmd_map[robot_id] = move(
                            game, motion_controller, robot_id, target_pos, oren, dribbling=True
                        )
                else:
                    ball_pos_for_clamp = Vector2D(ball.p.x, ball.p.y)
                    oren = ball_pos_for_clamp.angle_to(target_pos)
                    # Offset behind the ball (opposite the carry direction)
                    # so the robot's *dribbler* -- not its chassis centre --
                    # ends up on the ball; see _APPROACH_OFFSET's docstring.
                    # _clamp_to_field_or_ball keeps this reachable even when
                    # the ball itself sits right at (or, briefly, just past)
                    # the boundary -- see that helper's own docstring. A
                    # ball resting well out of bounds isn't chased at all in
                    # practice: strategy_runner.py teleports the ball onto
                    # designated_position (and re-pins it there until it's
                    # actually at rest -- see _TELEPORT_SETTLE_SPEED_MPS) at
                    # the moment BALL_PLACEMENT_* begins, precisely because
                    # "the robot can't physically retrieve an out-of-bounds
                    # ball in simulation" is a known, sim-only limitation
                    # handled at that layer, not here.
                    approach_dir = Vector2D(math.cos(oren), math.sin(oren))
                    approach = Vector2D(
                        ball_pos_for_clamp.x - approach_dir.x * self._APPROACH_OFFSET,
                        ball_pos_for_clamp.y - approach_dir.y * self._APPROACH_OFFSET,
                    )
                    target_for_move = _clamp_to_field_or_ball(approach, game, ball_pos_for_clamp)
                    self.blackboard.cmd_map[robot_id] = move(
                        game, motion_controller, robot_id, target_for_move, oren, dribbling=True
                    )
        return _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            exempt_robot_ids={placer_id},
        )


# ---------------------------------------------------------------------------
# BALL PLACEMENT — theirs
# ---------------------------------------------------------------------------


class BallPlacementTheirsStep:
    """Actively clear our robots away from the ball and target during their placement."""

    def update(self) -> None:
        return _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            designated_keep_dist=BALL_KEEP_OUT_DISTANCE,
        )


# ---------------------------------------------------------------------------
# PREPARE_KICKOFF — ours
# ---------------------------------------------------------------------------


class PrepareKickoffOursStep:
    """Positions robots for our kickoff.

    Goalkeeper (real ID from the referee packet) is exempt from formation and
    left for `GoalkeeperTactic`/the caller to handle. Lowest-ID non-keeper
    outfield robot approaches the ball at (0, 0); all other non-keeper robots
    move to own-half support positions outside the centre circle.
    """

    def update(self) -> None:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller
        ref = game.referee

        # Real goalkeeper ID from the referee packet — not assumed to be 0.
        # The keeper must not be chosen as the kicker, or the "kickoff"
        # becomes the keeper standing on the ball and then returning to its
        # line without ever touching it (observed: kernel matches with the
        # kicker = robot 0, back when 0 was assumed). Also exempt it from the
        # support formation entirely: unlike the penalty steps below, a
        # kickoff has no reason to pull the keeper off its line at all.
        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        robot_ids = sorted(rid for rid in game.friendly_robots.keys() if rid != keeper_id)
        if not robot_ids:
            return None
        kicker_id = robot_ids[0]

        # Kicker: approach from own-half side so the robot doesn't push the ball.
        own_half_sign = 1.0 if game.my_team_is_right else -1.0
        kicker_target = Vector2D(own_half_sign * (ROBOT_RADIUS + 0.03), 0.0)
        goal_x = _field_half_length(game) if not game.my_team_is_right else -_field_half_length(game)
        oren = math.atan2(0.0 - kicker_target.y, goal_x - kicker_target.x)
        self.blackboard.cmd_map[kicker_id] = move(game, motion_controller, kicker_id, kicker_target, oren)

        # Support robots: route through _clear_to_legal_positions so their paths
        # never cut across the keep-out zone even if they start on the enemy half.
        support_positions = _formation_positions(game, KICKOFF_SUPPORT_POSITION_RATIOS_OWN_HALF)
        support_idx = 0
        intended: dict[int, Vector2D] = {}
        for robot_id in robot_ids:
            if robot_id == kicker_id:
                continue
            intended[robot_id] = support_positions[support_idx % len(support_positions)]
            support_idx += 1

        return _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            clear_own_defense_area=True,
            clear_opp_defense_area=True,
            exempt_robot_ids={kicker_id, keeper_id},
            intended_targets=intended,
        )


# ---------------------------------------------------------------------------
# PREPARE_KICKOFF — theirs
# ---------------------------------------------------------------------------


class PrepareKickoffTheirsStep:
    """Moves all our non-keeper robots to own half, outside the centre circle,
    for the opponent kickoff. Goalkeeper (real ID from the referee packet) is
    exempt from formation, same as `PrepareKickoffOursStep`."""

    def update(self) -> None:
        game = self.blackboard.game
        ref = game.referee
        our_team_info = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
        keeper_id = our_team_info.goalkeeper

        positions = _formation_positions(game, KICKOFF_DEFENCE_POSITION_RATIOS_OWN_HALF)
        robot_ids = sorted(rid for rid in game.friendly_robots.keys() if rid != keeper_id)
        intended = {robot_id: positions[idx % len(positions)] for idx, robot_id in enumerate(robot_ids)}

        return _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            clear_own_defense_area=True,
            exempt_robot_ids={keeper_id},
            intended_targets=intended,
        )


# ---------------------------------------------------------------------------
# PREPARE_PENALTY — ours
# ---------------------------------------------------------------------------


class PreparePenaltyOursStep:
    """Positions robots for our penalty kick.

    Kicker (lowest non-keeper ID): moves to the penalty mark, faces goal.
    All others: stop on a line behind the penalty mark spread along y.

    Exact placement of non-kicker/non-keeper robots is a strategy decision
    and can be tuned here by the strategy team.
    """

    def update(self) -> None:
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
        intended = {}
        # The kicker waits just behind the ball, like DirectFreeOursStep's approach:
        # sent to the mark itself it drove onto the placed ball and shoved it off
        # the mark before NORMAL_START, and keep-out then voided the penalty.
        kicker_spot = Vector2D(penalty_mark.x - sign * RefereeGeometry._KICKER_APPROACH_M, 0.0)
        for robot_id in robot_ids:
            if robot_id == kicker_id:
                kicker = game.friendly_robots[robot_id]
                target = kicker_spot
                if game.ball is not None:
                    ball_pos = Vector2D(game.ball.p.x, game.ball.p.y)
                    target = _detour_around_circle(kicker.p, kicker_spot, ball_pos, ROBOT_RADIUS + BALL_RADIUS)
                self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, target, goal_oren)
            else:
                # Place behind the line, spread in y
                offset = (behind_idx - (len(robot_ids) - 1) / 2.0) * behind_y_step
                intended[robot_id] = Vector2D(behind_line_x, offset)
                behind_idx += 1

        # A teammate starting goal-side of the mark goes round the placed ball
        # rather than knocking it off the mark on the way to its line.
        _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            exempt_robot_ids={kicker_id},
            intended_targets=intended,
        )


# ---------------------------------------------------------------------------
# PREPARE_PENALTY — theirs
# ---------------------------------------------------------------------------


class PreparePenaltyTheirsStep:
    """Positions our robots for the opponent's penalty kick.

    Goalkeeper: moves to our goal line centre.
    All others: stop on a line behind the penalty mark spread along y.

    Exact placement of non-keeper robots is a strategy decision
    and can be tuned here by the strategy team.
    """

    def update(self) -> None:
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
        behind_line_x = opp_penalty_mark_x - sign * PENALTY_BEHIND_MARK_DISTANCE

        robot_ids = sorted(game.friendly_robots.keys())
        behind_idx = 0
        behind_y_step = PENALTY_LINE_Y_STEP_RATIO * _field_half_width(game)
        intended = {}

        for robot_id in robot_ids:
            if robot_id == keeper_id:
                # Keeper on own goal line, facing the incoming ball
                keeper_pos = Vector2D(our_goal_x, 0.0)
                self.blackboard.cmd_map[robot_id] = move(
                    game, motion_controller, robot_id, keeper_pos, math.pi if game.my_team_is_right else 0.0
                )
            else:
                offset = (behind_idx - (len(robot_ids) - 1) / 2.0) * behind_y_step
                intended[robot_id] = Vector2D(behind_line_x, offset)
                behind_idx += 1

        # The line is on the far side of the ball from our goal, so a defender
        # starting goal-side must go round the ball, not through the keep-out circle.
        _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            exempt_robot_ids={keeper_id},
            intended_targets=intended,
        )


# ---------------------------------------------------------------------------
# DIRECT_FREE — ours
# ---------------------------------------------------------------------------


class DirectFreeOursStep:
    """Positions our robots for our direct free kick.

    The robot closest to the ball becomes the kicker and approaches from the
    field-inward side so it cannot push the ball out of bounds.
    All other robots stop in place.
    """

    # How close the robot center should get behind the ball before turning/kicking.
    _APPROACH_OFFSET = RefereeGeometry._KICKER_APPROACH_M
    _APPROACH_READY_DISTANCE = 0.04
    _KICK_READY_DISTANCE = 0.16
    _FACE_READY_ANGLE = 0.18

    # Robot-body clearance only, not the full OPPONENT_DEFENSE_AREA_KEEP_
    # DISTANCE (0.25m) `StopStep`/`_clear_to_legal_positions` use for general
    # restart positioning. `DefenseAreaRule`'s actual foul condition
    # (`in_yellow_defense`/`in_blue_defense` in defense_area_rule.py) is a
    # strict boundary test, not a keep-distance — and the *ball* itself can
    # legally sit arbitrarily close to (or, awarded by other rules, right at)
    # the box edge, since `legal_restart_position`'s own placement logic only
    # keeps the ball OPPONENT_DEFENSE_AREA_KEEP_DISTANCE clear, not the
    # kicker. Using the full 0.25m keep distance here computes an approach
    # point the kicker can never close to within `_KICK_READY_DISTANCE`
    # (0.16m) whenever the ball itself sits within 0.25m of the box —
    # confirmed live, 2026-09-05: a full-length match's DIRECT_FREE_YELLOW
    # kicker converged to exactly the ball-relative offset a fix like that
    # would produce and then held there for the rest of a 600s match, unable
    # to ever reach kicking range. Robot-radius clearance keeps the kicker's
    # body legally outside the box while staying reachable.
    _KICKER_DEFENSE_AREA_CLEARANCE = ROBOT_RADIUS

    # Metres a challenger must be closer than the current kicker before the
    # role actually flips — same shape/value as pass_and_shoot.py's
    # _REASSIGN_MARGIN_M. Without this, a naive `min(..., key=distance)`
    # recomputed fresh every tick flips the "closest" robot on ordinary
    # rsim position noise whenever two-plus robots sit near-equidistant from
    # the ball (observed: three robots within 2cm of each other, ~9 flips/
    # second, 243 in 27s straight-line trace). Every flip resets the new
    # kicker's approach from a standing start (the old kicker becomes
    # `empty_command()` immediately), so the role never holds long enough
    # for any robot to actually close the distance — this was the real
    # mechanism behind the DIRECT_FREE stall traced in
    # clear_danger_vs_clear_press_plus (roadmap item 15), not the
    # multi-robot planner congestion first suspected from replay data alone.
    _KICKER_REASSIGN_MARGIN_M = 0.3

    def __init__(self):
        self._kicker_sticky = Sticky[int](margin=self._KICKER_REASSIGN_MARGIN_M)

    @staticmethod
    def _angle_error(current: float, target: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _nearest_enemy_to_ball(self, game, ball_pos: Vector2D):
        if not game.enemy_robots:
            return None, None
        enemy_id = min(game.enemy_robots, key=lambda rid: game.enemy_robots[rid].p.distance_to(ball_pos))
        return enemy_id, game.enemy_robots[enemy_id]

    def _kick_target_enemy(self, game, ball_pos: Vector2D):
        if 1 in game.enemy_robots:
            return 1, game.enemy_robots[1]
        return self._nearest_enemy_to_ball(game, ball_pos)

    def update(self) -> None:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller
        ball = game.ball

        if not ball:
            _all_stop(self.blackboard)
            return

        distances = {rid: game.friendly_robots[rid].p.distance_to(ball.p) for rid in game.friendly_robots}
        kicker_id = self._kicker_sticky.update(list(distances), score_fn=lambda rid: -distances[rid])

        for robot_id in game.friendly_robots:
            if robot_id == kicker_id:
                robot = game.friendly_robots[robot_id]
                ball_pos = Vector2D(ball.p.x, ball.p.y)
                distance_to_ball = robot.p.distance_to(ball_pos)
                _, enemy = self._kick_target_enemy(game, ball_pos)

                if enemy is None:
                    target_oren = robot.p.angle_to(ball_pos)
                else:
                    target_oren = ball_pos.angle_to(enemy.p)

                kick_dir = Vector2D(math.cos(target_oren), math.sin(target_oren))
                approach = Vector2D(
                    ball_pos.x - kick_dir.x * self._APPROACH_OFFSET,
                    ball_pos.y - kick_dir.y * self._APPROACH_OFFSET,
                )
                approach = _clamp_to_field_or_ball(approach, game, ball_pos)
                # Same ball-relative-distance-preserving guard as
                # _clamp_to_field_or_ball above, for the opponent's defense
                # area instead of the field boundary — the ball can sit just
                # as close to the box edge as it can to the sideline (see
                # _KICKER_DEFENSE_AREA_CLEARANCE's comment), and pushing the
                # kicker's target strictly outside the box regardless of the
                # ball's own position creates the same "target permanently
                # farther from the ball than kick range" deadlock.
                projected = _project_outside_opp_defense_area(game, approach, self._KICKER_DEFENSE_AREA_CLEARANCE)
                if (projected - ball_pos).mag() <= (approach - ball_pos).mag():
                    approach = projected
                distance_to_approach = robot.p.distance_to(approach)
                face_error = self._angle_error(robot.orientation, target_oren)

                if distance_to_approach > self._APPROACH_READY_DISTANCE:
                    # Round the ball, not through it, to an approach point on its far side:
                    # driving through pushed free kicks placed 0.25 m inside the line back
                    # onto it (tournament_20260927_223257).
                    waypoint = _detour_around_circle(robot.p, approach, ball_pos, ROBOT_RADIUS + BALL_RADIUS)
                    self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, waypoint, target_oren)
                elif abs(face_error) > self._FACE_READY_ANGLE:
                    self.blackboard.cmd_map[robot_id] = turn_on_spot(
                        game, motion_controller, robot_id, target_oren, dribbling=False
                    )
                elif distance_to_ball > self._KICK_READY_DISTANCE:
                    self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, approach, target_oren)
                else:
                    self.blackboard.cmd_map[robot_id] = empty_command(False)
            else:
                self.blackboard.cmd_map[robot_id] = empty_command(False)


# ---------------------------------------------------------------------------
# DIRECT_FREE — theirs
# ---------------------------------------------------------------------------


class DirectFreeTheirsStep:
    """Actively clear our robots out of the ball keep-out radius."""

    def update(self) -> None:
        return _clear_to_legal_positions(
            self.blackboard,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
        )


# ---------------------------------------------------------------------------
# Helper: resolve bilateral commands
# ---------------------------------------------------------------------------


def is_our_command(command: RefereeCommand, our_command: RefereeCommand, their_command: RefereeCommand) -> bool:
    """Not used directly — bilateral resolution is done in tree.py via command sets."""
    return command == our_command
