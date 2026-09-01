"""Plain-function geometry/condition helpers shared by pass-and-attack tactics.

Ported from `utama_strategy.functional.skills`, with one change: the
original imported `find_best_shot` from `utama_strategy.utils.score_goal_utils`,
a Strategy-repo-private near-duplicate of Core's own `_find_best_shot`
(same ray-casting/shadow algorithm, forked at some point — see that
module's docstring). Since this code now lives in Core, it uses Core's
own `_find_best_shot` directly instead of carrying the duplicate forward.

Each function is `(game, ...) -> value`: no blackboard, no py_trees
Status, no hidden state — with one deliberate exception, `has_ball`'s
`_POSSESSION_STATE`, see that function's docstring for why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.skills.src.score_goal import (  # noqa: F401  (re-exported)
    _find_best_shot,
    is_goal_blocked,
)

ORIENTATION_TOLERANCE_RAD = 0.05

# `has_ball(visual=True)`'s dribbler-relative acquire/release box — see that
# function's docstring for the derivation. Measured against rsim's own
# `isTouchingBall()`-backed `robot.has_ball` (`vendor/rSim/src/robosim/
# sslrobot.cpp:127-144`, config constants in `sslconfig.h`:
# `distanceCenterKicker=0.081`, `kickerWidth=0.080`) across two replays
# (`switch_of_play_vs_default.pkl`, `tiki_taka_vs_counter_press.pkl`):
# every real-sensor-True tick had the ball's chassis-frame forward offset in
# [0.091, 0.112]m and lateral offset within ±0.043m. `_ACQUIRE_*` below adds
# a small margin around that envelope (enough to say "yes" slightly before
# rsim's own hold hinge would latch, since this fallback must also work for
# robots/situations with no real sensor at all — enemy inference, real
# hardware); `_RELEASE_*` is wider again, for the hysteresis band.
_ACQUIRE_FORWARD_MIN = 0.0
_ACQUIRE_FORWARD_MAX = 0.14
_ACQUIRE_LATERAL_MAX = 0.05
_RELEASE_FORWARD_MAX = 0.17
_RELEASE_LATERAL_MAX = 0.065

# Per-robot "did the last tick's visual has_ball read True" state, keyed by
# robot_id only (matching `shielding.py`'s `_COMMITTED_ROBOTS` — friendly-only
# callers, so no team key needed). This is a deliberate departure from this
# module's "no hidden state" rule, for the same reason `shielding.py` made
# the same departure (see its `_COMMITTED_ROBOTS` docstring): a bare
# stateless threshold cannot implement hysteresis, because the decision must
# depend on which side of the band the caller last committed to, and a bare
# threshold right at the dribbler-box boundary was confirmed to chatter
# tick-to-tick (this fallback's whole reason for existing — see the
# docstring's flicker paragraph). Never grows unboundedly: at most one entry
# per robot ID actually in play.
_POSSESSION_STATE: dict[int, bool] = {}


def reset_possession_state(robot_id: int) -> None:
    """Forget any held-ball hysteresis state for `robot_id`.

    Call this wherever a robot's ball possession is being reset for an
    unrelated reason (role reassignment, a fresh acquisition attempt after
    losing the ball to an enemy) so a stale "was holding" commitment doesn't
    widen the acquire box to the release box on the next tick. Safe to call
    even if no state is held (no-op). Mirrors `shielding.reset_shield_state`.
    """
    _POSSESSION_STATE.pop(robot_id, None)


def has_ball(game: Game, robot_id: int, visual: bool = False, capture_distance: float = _ACQUIRE_FORWARD_MAX) -> bool:
    """Friendly-only: `robot.has_ball` (the non-visual path) is our own robots'
    IR contact sensor. There is no equivalent real sensor for enemy robots —
    in sim, rsim's physics engine happens to expose ground-truth contact for
    both teams, but tactic decision-making must behave the same in sim and on
    real hardware, where we will never have an opponent's IR reading. Enemy
    possession must be inferred visually (`visual=True`) instead; don't add a
    team switch here for tactic code. (The referee's own bookkeeping, which
    reads `Robot.has_ball` directly off frame objects rather than through this
    helper, is the one place sim's ground-truth enemy contact data is used.)

    `visual=True` used to be a plain circle around the robot's chassis center
    (`distance_to(ball) < capture_distance`) — no orientation or lateral
    check at all, so it read True for a ball merely beside the chassis, or in
    front of a robot facing away from it (ball chassis-adjacent but nowhere
    near the dribbler). Measured directly against `robot.has_ball` (the real
    sensor) tick-by-tick across two replays: the old circle false-positived
    on 6.43% and 1.49% of friendly-robot ticks, roughly three-quarters of
    which were >30 degrees off the ball's true bearing (facing-away or
    lateral cases), not just marginal boundary chatter. This is the "robot
    thought it had the ball but didn't" failure that broke grab-ball-and-go.

    Now shaped like rsim's own kicker box instead of a chassis-centered
    circle: `capture_distance` is a *forward* depth from the chassis center
    along `robot.orientation` (approximating the kicker/dribbler position,
    ~0.081m forward per rsim's `distanceCenterKicker` — see module-level
    `_ACQUIRE_FORWARD_MAX`/`_ACQUIRE_LATERAL_MAX` for the measured box), with
    a separate, narrower lateral half-width, rather than a symmetric radius.
    Re-measured after this change (box geometry plus the hysteresis
    described below — i.e. the actual shipped behaviour): false positives
    dropped to 1.21% and 0.98% on the same two replays (from 6.43%/1.49%)
    with zero new false negatives (every real-sensor True tick's ball offset
    already sat well inside the new box, per the measured envelope above) —
    see `docs/investigation_ball_contact_orientation_divergence.md`'s
    sibling investigation for the kicker-box geometry this approximates, and
    this function's own git history / commit message for the exact
    before/after table.

    Hysteresis (commit/release) is layered on top of that shaped box, via
    module-level `_POSSESSION_STATE`, to address the flicker half of the
    same complaint: once a robot's visual read goes True, it keeps reading
    True until the ball leaves the wider `_RELEASE_*` box, not just the
    tighter `_ACQUIRE_*` one — the same commit/release pattern
    `shielding.py`'s `_COMMIT_RANGE`/`_RELEASE_RANGE` uses, for the same
    reason (a bare threshold right at a boundary chatters when per-tick
    simulator jitter straddles it). Measured cost of this hysteresis: in the
    worst observed case across both replays, the visual read stayed True for
    up to 15 ticks (~0.25s at 60Hz) after the real sensor had already gone
    False — bounded and comparable to other deliberate grace periods already
    in this codebase (e.g. `_pass_and_score.py`'s
    `_SETUP_BALL_LOSS_GRACE_TICKS=10`), not a runaway "stuck thinking it has
    the ball forever" failure mode.

    `capture_distance` keeps its old name/position (positional- and
    keyword-compatible) and still means "how far the ball can be and still
    count as possessed" for existing callers — it now bounds the forward
    depth of the acquire box instead of a circle's radius, tightened from the
    old 0.15m default to 0.14m per the measured envelope above (still
    comfortably above `ROBOT_RADIUS + BALL_RADIUS ≈ 0.1115m` contact
    distance). Passing a custom `capture_distance` widens/narrows only the
    forward reach of the *acquire* box; the release box and lateral
    half-widths are not parameterized (no caller needs that yet — add it if
    one does).
    """
    robot = game.friendly_robots[robot_id]
    if not visual:
        return bool(robot.has_ball)

    ball = game.ball.p.to_2d()
    dx = ball.x - robot.p.x
    dy = ball.y - robot.p.y
    cos_o, sin_o = math.cos(robot.orientation), math.sin(robot.orientation)
    # Rotate the robot->ball vector into the robot's own frame: `forward` is
    # the component along `robot.orientation` (positive = in front of the
    # chassis, toward the dribbler), `lateral` is the perpendicular component.
    forward = dx * cos_o + dy * sin_o
    lateral = -dx * sin_o + dy * cos_o

    already_had_it = _POSSESSION_STATE.get(robot_id, False)
    # A caller-supplied `capture_distance` only ever sets the *acquire*
    # forward reach (see docstring) — once already committed (hysteresis),
    # the release box is fixed, matching `shielding.py`'s pattern of not
    # letting the acquire-side parameter reach into the release band.
    forward_max = _RELEASE_FORWARD_MAX if already_had_it else capture_distance
    lateral_max = _RELEASE_LATERAL_MAX if already_had_it else _ACQUIRE_LATERAL_MAX

    result = (_ACQUIRE_FORWARD_MIN <= forward <= forward_max) and (abs(lateral) <= lateral_max)
    _POSSESSION_STATE[robot_id] = result
    return result


def at_target(game: Game, robot_id: int, target: Vector2D, tolerance: float = 0.08) -> bool:
    robot = game.friendly_robots[robot_id]
    return robot.p.distance_to(target) <= tolerance


def oriented_towards(
    game: Game, robot_id: int, target_orientation: float, tolerance: float = ORIENTATION_TOLERANCE_RAD
) -> bool:
    robot = game.friendly_robots[robot_id]
    diff = (target_orientation - robot.orientation + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) <= tolerance


def clamp_to_field(position: Vector2D, game: Game, margin: float = 0.3) -> Vector2D:
    field = game.field
    x = max(-field.half_length + margin, min(field.half_length - margin, position.x))
    y = max(-field.half_width + margin, min(field.half_width - margin, position.y))
    return Vector2D(x, y)


def intercept_point(
    game: Game,
    passer_id: int,
    receiver_id: int,
    min_intercept_distance: float = 0.5,
) -> tuple[Vector2D, float]:
    """Returns (intercept_position, intercept_orientation)."""
    passer = game.friendly_robots[passer_id]
    receiver = game.friendly_robots[receiver_id]
    ball_pos = game.ball.p.to_2d()

    trajectory_direction = Vector2D(math.cos(passer.orientation), math.sin(passer.orientation))
    projection_t = (receiver.p - ball_pos).dot(trajectory_direction)
    intercept_distance = max(projection_t, min_intercept_distance)
    intercept_position = clamp_to_field(ball_pos + trajectory_direction * intercept_distance, game)

    intercept_orientation = math.atan2(ball_pos.y - receiver.p.y, ball_pos.x - receiver.p.x)
    return intercept_position, intercept_orientation


def enemy_positions(game: Game) -> list[Vector2D]:
    return [enemy.p for enemy in game.enemy_robots.values() if enemy is not None]


_LOOSE_BALL_SPEED = 0.3  # m/s — matches goalkeep.py's own-box retrieval threshold
_LOOSE_BALL_CONTEST_RANGE = 1.5  # metres — matches PressAndContainTactic's own _PRESS_RANGE


def ball_is_loose(game: Game, contest_range: float = _LOOSE_BALL_CONTEST_RANGE) -> bool:
    """True when the ball is sitting dead with no enemy nearby to contest it.

    Every "defensive shape" tactic (`DefenseTactic`, `BlockShapeTactic`,
    `ShadowAndMarkTactic`) positions purely off the ball-to-goal shot angle or
    off enemy robots, and never actually approaches the ball itself —
    correct when an enemy has or is about to have it (that's what shadowing
    prepares for), but wrong once the ball is abandoned altogether: nothing
    in any of those tactics' geometry ever changes for a ball with no owner,
    so it just sits wherever it stopped, for the rest of the match, even deep
    in a defender's own corner with a friendly robot standing a metre away.
    `PressAndContainTactic` already opts out of exactly this case via its own
    `applicable()` (`_PRESS_RANGE`) rather than stepping on it — this uses
    the same range so "no one is contesting it" means the same thing across
    every defensive tactic that checks it.

    Deliberately does not check which team's corner the ball is in, or
    distance from any particular robot — that's for the caller (typically:
    "is my own nearest assigned robot closer to the ball than
    `_LOOSE_BALL_CONTEST_RANGE`, and if so send it to fetch the ball instead
    of holding its normal shape this tick").
    """
    if game.ball is None:
        return False
    ball_speed = (game.ball.v.x**2 + game.ball.v.y**2) ** 0.5
    if ball_speed >= _LOOSE_BALL_SPEED:
        return False
    ball_pos = game.ball.p.to_2d()
    for enemy in game.enemy_robots.values():
        if enemy is not None and enemy.p.distance_to(ball_pos) <= contest_range:
            return False
    return True


def _distance_to_segment(point: Vector2D, start: Vector2D, end: Vector2D) -> float:
    segment = end - start
    segment_len_sq = segment.dot(segment)
    if segment_len_sq <= 1e-12:
        return point.distance_to(start)
    projection = max(0.0, min(1.0, (point - start).dot(segment) / segment_len_sq))
    closest = start + segment * projection
    return point.distance_to(closest)


def segment_clearance(start: Vector2D, end: Vector2D, obstacles: list[Vector2D]) -> float:
    """Minimum distance from any obstacle to the start->end segment. 10.0 if none."""
    if not obstacles:
        return 10.0
    return min(_distance_to_segment(obstacle, start, end) for obstacle in obstacles)


def segment_blocked(
    start: Vector2D, end: Vector2D, obstacles: list[Vector2D], clearance: float = ROBOT_RADIUS + 0.15
) -> bool:
    return segment_clearance(start, end, obstacles) <= clearance


def enemy_goal_line(game: Game) -> tuple[float, float, float]:
    goal_line = game.field.enemy_goal_line
    goal_x = float(goal_line[0][0])
    goal_y1 = min(float(goal_line[0][1]), float(goal_line[1][1]))
    goal_y2 = max(float(goal_line[0][1]), float(goal_line[1][1]))
    return goal_x, goal_y1, goal_y2


def in_own_defense_area(game: Game, point: Vector2D) -> bool:
    """True if `point` is inside our own defense area.

    Uses `field.my_defense_area` — the same geometry the CustomReferee's
    `DefenseAreaRule` derives from (`half_defense_area_depth`/`width`), so a
    tactic deciding legality by this check agrees with the referee.
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    goal_x = game.field.my_goal_line[0][0]
    half_width = abs(float(defense_area[0][1]))
    x_inside = (point.x - front_x) * (goal_x - front_x) >= 0.0
    return x_inside and abs(point.y) <= half_width


def ball_in_own_defense_area(game: Game) -> bool:
    """True if the ball center is inside our own defense area (see `in_own_defense_area`)."""
    return in_own_defense_area(game, game.ball.p.to_2d())


def in_enemy_defense_area(game: Game, point: Vector2D) -> bool:
    """True if `point` is inside the enemy's defense area. Mirror of `in_own_defense_area`."""
    defense_area = game.field.enemy_defense_area
    front_x = float(defense_area[1][0])
    goal_x = game.field.enemy_goal_line[0][0]
    half_width = abs(float(defense_area[0][1]))
    x_inside = (point.x - front_x) * (goal_x - front_x) >= 0.0
    return x_inside and abs(point.y) <= half_width


def ball_in_enemy_defense_area(game: Game) -> bool:
    """True if the ball center is inside the enemy's defense area (see `in_enemy_defense_area`).

    An attacker may not enter to retrieve it during active play — see
    `FastPathPlanner._enemy_defense_area_retrieval_exempt`'s docstring, which
    only lifts that keep-out during a stoppage restart, never during
    NORMAL_START/FORCE_START. A tactic must check this itself and hold
    rather than call `go_to_ball` at the literal ball position, or the
    planner clamps every target back to the boundary and the robot
    oscillates along it indefinitely (found live: `LeadAndSupportTactic`'s
    sole leader orbiting the enemy box edge for 20+ seconds while the enemy
    goalkeeper held the ball inside it, see `docs/investigation_*.md`).
    """
    return in_enemy_defense_area(game, game.ball.p.to_2d())


def clamp_outside_own_defense_area(game: Game, point: Vector2D, margin: float = 2.0 * ROBOT_RADIUS + 0.05) -> Vector2D:
    """Clamp a target point to just outside our own defense area's front edge.

    The `DefenseAreaRule` fouls any outfield robot entering the area (the
    keeper owns the box), so tactics that route robots near their own goal —
    shot-line defenders, carriers chasing a loose ball — must never target
    inside it. Keep-out is enforced on x with one robot-diameter margin;
    the y-coordinate is preserved unchanged (the area only spans
    `half_defense_area_width`, so a clamped x alone is already outside).
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    sign = 1.0 if game.my_team_is_right else -1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def own_defense_area_exit_point(game: Game, at_y: float, margin: float = 2.0 * ROBOT_RADIUS + 0.05) -> Vector2D:
    """A hold point just outside our own defense area's front edge at `at_y`.

    For a defender/carrier that needs to stand near a ball that is inside
    our own area (which outfield robots may not enter): hold the edge
    closest to the ball, with y clamped inside the area's width so the
    point is the nearest legal standing spot.
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    half_width = abs(float(defense_area[0][1]))
    sign = 1.0 if game.my_team_is_right else -1.0
    exit_x = front_x - sign * margin
    y = max(-(half_width - margin), min(half_width - margin, at_y))
    return Vector2D(exit_x, y)


def clamp_outside_enemy_defense_area(
    game: Game, point: Vector2D, margin: float = 2.0 * ROBOT_RADIUS + 0.05
) -> Vector2D:
    """Clamp a target point to just outside the enemy's defense area front edge.

    `DefenseAreaRule` fouls attacker encroachment into the enemy box too (not
    just our own — `attacker_infringement=True` is the referee default), so
    any attacking tactic that scripts a target close to the enemy goal
    (overload/relay finishing runs, decoy lures, switch-of-play runners) must
    clamp it the same way `clamp_outside_own_defense_area` clamps defensive
    targets. Mirror image of that function: the enemy goal is on the
    opposite side from ours, so the clamp direction is `-sign` instead of
    `sign`.
    """
    defense_area = game.field.enemy_defense_area
    front_x = float(defense_area[1][0])
    sign = -1.0 if game.my_team_is_right else 1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def enemy_defense_area_hold_point(game: Game, at_y: float, margin: float = 2.0 * ROBOT_RADIUS + 0.05) -> Vector2D:
    """A hold point just outside the enemy's defense area front edge at `at_y`.

    Mirror of `own_defense_area_exit_point`, for an attacker that needs to
    wait near a ball resting inside the enemy's area (which it may not enter
    during active play — see `ball_in_enemy_defense_area`) instead of
    endlessly re-targeting the ball itself.
    """
    defense_area = game.field.enemy_defense_area
    front_x = float(defense_area[1][0])
    half_width = abs(float(defense_area[0][1]))
    sign = -1.0 if game.my_team_is_right else 1.0
    exit_x = front_x - sign * margin
    y = max(-(half_width - margin), min(half_width - margin, at_y))
    return Vector2D(exit_x, y)


def find_best_shot(
    point: Vector2D,
    enemy_robots: list,
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
    prev_best_shot_y: Optional[float] = None,
    switch_margin: float = 0.0,
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Thin passthrough to Core's own shadow/ray-casting shot finder.

    `prev_best_shot_y`/`switch_margin`: optional hysteresis, forwarded
    unchanged — see `_find_best_shot`'s docstring in `skills/src/score_goal.py`.
    Both default to no-hysteresis (today's exact behaviour); a caller opts in
    by passing its own previously-chosen shot y (typically from its `mem`).
    """
    return _find_best_shot(point, enemy_robots, goal_x, goal_y1, goal_y2, prev_best_shot_y, switch_margin)


_NO_SHOT_STRAFE_STEP = 0.6  # metres of lateral reposition per tick while hunting for a lane


def no_shot_reposition_target(
    carrier_pos: Vector2D, enemy_robots: list, goal_x: float, goal_y1: float, goal_y2: float, field_half_width: float
) -> Vector2D:
    """Where to dribble to when `find_best_shot` returns no lane at all.

    Every `find_best_shot(...) -> (None, None)` call site used to respond by
    freezing in place (`empty_command(dribbler_on=True)`) — correct only if
    the blocker is about to move on its own. Against a stationary keeper
    covering the whole goal from close range (the common case once a carrier
    reaches the goal mouth) nothing about that position ever changes, so the
    freeze is permanent: this was the actual mechanism behind a `clear_danger`
    vs `low_block` match pinning 0-0 for 48 straight seconds (see
    `docs/strategies.md`'s "Known open bugs" and
    `docs/investigation_default_vs_lowblock_stalemate.md` for the sibling
    pin this shares its shape with, at the ball-approach stage rather than
    here at the finishing stage).

    Fix: step laterally, away from the nearest blocker's side of the goal,
    which changes every enemy's shadow angle (`_ray_casting`) enough that a
    gap reliably opens within a few strafes — cheaper than reasoning about
    the blockers' shadows directly, and correct for the same reason standing
    still is wrong: motion is the only thing that changes this calculation's
    inputs. Falls back to sliding toward goal-center if there are no
    enemies at all (shouldn't happen when `find_best_shot` just returned
    `None`, but keeps this total).

    Clamped-at-the-sideline case (found live, 2026-09-01 tournament stuck-
    match investigation): a carrier already within `margin` of the field's
    y-boundary has its away-from-blocker step clamped right back to
    (near-)its own current position — the intended 0.6m strafe collapses to
    a few millimetres, which recreates the exact "nothing about this
    position ever changes" freeze this function exists to avoid, just from
    a boundary clamp instead of a stationary keeper. Traced live: a carrier
    pinned at y=-2.697 (half_width=3.0, margin=0.3, clamp at -2.7) computing
    a step toward -2.7 moved a net 0.3cm and then repeated the identical
    clamped target forever. If the preferred direction is clamped away to
    within `_NO_SHOT_STRAFE_STEP / 2` of the current position, strafe the
    other way instead — the opposite direction always has room, since a
    field can't be narrower than one strafe step in total width.
    """
    if enemy_robots:
        nearest = min(enemy_robots, key=lambda e: carrier_pos.distance_to(e))
        # Step away from whichever side of us the nearest blocker sits on; a
        # blocker dead level (rare — tie only matters for direction, not
        # whether to move) breaks toward +y arbitrarily.
        away_sign = 1.0 if nearest.y <= carrier_pos.y else -1.0
    else:
        goal_mid_y = (goal_y1 + goal_y2) / 2.0
        away_sign = 1.0 if carrier_pos.y >= goal_mid_y else -1.0
    margin = 0.3

    def _clamped_target_y(sign: float) -> float:
        return max(
            -field_half_width + margin, min(field_half_width - margin, carrier_pos.y + sign * _NO_SHOT_STRAFE_STEP)
        )

    target_y = _clamped_target_y(away_sign)
    if abs(target_y - carrier_pos.y) < _NO_SHOT_STRAFE_STEP / 2:
        # The preferred direction was clamped away to a near-zero step —
        # flip direction rather than silently freezing against the boundary.
        target_y = _clamped_target_y(-away_sign)
    return Vector2D(carrier_pos.x, target_y)


@dataclass(frozen=True)
class PassSetupScore:
    passer_position: Vector2D
    receiver_position: Vector2D
    score: float


def score_pass_setup(
    game: Game,
    passer_position: Vector2D,
    receiver_position: Vector2D,
    min_pass_distance: float = 0.7,
    min_robot_clearance: float = ROBOT_RADIUS + 0.15 + 0.2,
) -> Optional[PassSetupScore]:
    """Score a candidate (passer_position, receiver_position) setup pair.

    Returns None if the pair is infeasible (too close together, blocked
    pass, no open shot, blocked shot, or too close to a friendly robot).
    """
    pass_distance = passer_position.distance_to(receiver_position)
    if pass_distance < min_pass_distance:
        return None

    friendly_positions = [robot.p for robot in game.friendly_robots.values() if robot is not None]
    for robot_pos in friendly_positions:
        if passer_position.distance_to(robot_pos) < min_robot_clearance:
            return None
        if receiver_position.distance_to(robot_pos) < min_robot_clearance:
            return None

    enemies = enemy_positions(game)
    if segment_blocked(passer_position, receiver_position, enemies):
        return None

    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, largest_gap = find_best_shot(
        receiver_position, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2
    )
    if best_shot_y is None or largest_gap is None:
        return None

    shot_target = Vector2D(goal_x, best_shot_y)
    if segment_blocked(receiver_position, shot_target, enemies):
        return None

    pass_clearance = segment_clearance(passer_position, receiver_position, enemies)
    shot_gap = largest_gap[1] - largest_gap[0]
    distance_to_goal_ratio = abs(receiver_position.x - goal_x) / max(2.0 * abs(goal_x), 1e-6)
    score = shot_gap + 0.3 * pass_clearance - 0.2 * distance_to_goal_ratio - 0.03 * pass_distance

    return PassSetupScore(passer_position=passer_position, receiver_position=receiver_position, score=score)
