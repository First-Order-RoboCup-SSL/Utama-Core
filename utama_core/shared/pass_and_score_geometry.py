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
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig,
)
from utama_core.skills.src.score_goal import (  # noqa: F401  (re-exported)
    _find_best_shot,
    is_goal_blocked,
)

ORIENTATION_TOLERANCE_RAD = 0.05

# `clamp_outside_own_defense_area`/`clamp_outside_enemy_defense_area` (and
# their `*_hold_point` siblings) previously defaulted to `2*ROBOT_RADIUS+0.05
# = 0.23m` past `front_x`, the box's true edge. But `FastPathPlanner` doesn't
# draw its obstacle segment at `front_x` -- it draws it `OPPONENT_DEFENSE_
# AREA_KEEP_DISTANCE` (0.25m) further out (see `_enemy_defense_rect`'s
# `margin` and `_get_obstacles`), and then refuses to route any target's
# *carrot* closer than another `OBSTACLE_CLEARANCE` (`ROBOT_DIAMETER *
# CLEARANCE_MULTIPLIER` = 0.27m at full clearance) to that segment. A margin
# of 0.23m from `front_x` lands *inside* the already-offset obstacle line,
# not past it -- the carrot converges on the clearance ring and never closes
# the last stretch to the actual (perfectly legal) target, exactly the
# failure `RefereeGeometry.legal_restart_position`'s `_PLANNER_CLEARANCE_
# BUFFER_M` exists to avoid for restart positions (that fix adds its buffer
# on top of the same `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE`-based `keep_dist`
# call sites already pass it). Traced live, 2026-09-05
# (tiki_taka_vs_zone_fluid_RK, COMMITTED_FROZEN, 10.3s+ during FORCE_START):
# `GiveAndGoTactic`'s in-flight pass computed a receiver intercept point
# that `clamp_outside_enemy_defense_area` clamped to x=-3.27 (only 0.02m
# past the planner's obstacle segment at x=-3.25 = -3.5 + 0.25) -- the
# receiver's carrot stuck at x=-3.0232, `at_target`'s 0.08m tolerance never
# satisfied, so the pass handshake never completed and `GiveAndGoTactic`
# re-picked the same unreachable receiver every 4s (`_MAX_HOP_TICKS`)
# forever. Set past `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE + OBSTACLE_
# CLEARANCE` (0.25 + 0.27 = 0.52m) with the same margin above that total
# `RefereeGeometry` keeps above its own equivalent figure.
_DEFENSE_AREA_CLAMP_MARGIN = OPPONENT_DEFENSE_AREA_KEEP_DISTANCE + fastpathplanningconfig.OBSTACLE_CLEARANCE + 0.05

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
# (team, robot_id): both teams' strategies run in one process with the same
# robot ids, and keyed by id alone one team's possession widened the other's
# box. A new match clears it (`StrategyRunner` calls `reset_possession_state()`),
# since a round-robin worker plays many matches. This is a deliberate departure from this
# module's "no hidden state" rule, for the same reason `shielding.py` made
# the same departure (see its `_COMMITTED_ROBOTS` docstring): a bare
# stateless threshold cannot implement hysteresis, because the decision must
# depend on which side of the band the caller last committed to, and a bare
# threshold right at the dribbler-box boundary was confirmed to chatter
# tick-to-tick (this fallback's whole reason for existing — see the
# docstring's flicker paragraph). Never grows unboundedly: at most one entry
# per robot ID actually in play.
_POSSESSION_STATE: dict[tuple[bool, int], bool] = {}


def reset_possession_state(robot_id: Optional[int] = None) -> None:
    """Forget any held-ball hysteresis state for `robot_id` (either team's), or for every
    robot when None.

    Call this wherever a robot's ball possession is being reset for an
    unrelated reason (role reassignment, a fresh acquisition attempt after
    losing the ball to an enemy) so a stale "was holding" commitment doesn't
    widen the acquire box to the release box on the next tick. Safe to call
    even if no state is held (no-op). Mirrors `shielding.reset_shield_state`.
    """
    if robot_id is None:
        _POSSESSION_STATE.clear()
    for team_is_yellow in (True, False):
        _POSSESSION_STATE.pop((team_is_yellow, robot_id), None)


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

    key = (game.my_team_is_yellow, robot_id)
    already_had_it = _POSSESSION_STATE.get(key, False)
    # A caller-supplied `capture_distance` only ever sets the *acquire*
    # forward reach (see docstring) — once already committed (hysteresis),
    # the release box is fixed, matching `shielding.py`'s pattern of not
    # letting the acquire-side parameter reach into the release band.
    forward_max = _RELEASE_FORWARD_MAX if already_had_it else capture_distance
    lateral_max = _RELEASE_LATERAL_MAX if already_had_it else _ACQUIRE_LATERAL_MAX

    result = (_ACQUIRE_FORWARD_MIN <= forward <= forward_max) and (abs(lateral) <= lateral_max)
    _POSSESSION_STATE[key] = result
    return result


# `ExcessiveDribblingRule` fouls a carry of more than 1.0m; the margin covers
# the ball swinging round the dribbler while the carrier turns to kick.
CARRY_LIMIT_M = 0.8
# Braking deceleration for the stopping-distance term in `carry_exhausted`.
# Measured, not `MAX_ACCELERATION` (4.0): a DecoyOverload carrier in rsim
# braked 1.45 -> 0.6m/s in 0.3s (~2.9m/s^2) with the ball on the dribbler.
_CARRY_BRAKE_DECEL_MPS2 = 2.5


def carry_origin(game: Game, robot_id: int, prev_origin: Optional[Vector2D]) -> Optional[Vector2D]:
    """Where `robot_id`'s current dribble began: the ball position when
    `robot.has_ball` last went True, the same signal and point
    `ExcessiveDribblingRule` measures from. None while not holding the ball.
    Call once per tick with the previous return value."""
    robot = game.friendly_robots.get(robot_id)
    if robot is None or not robot.has_ball or game.ball is None:
        return None
    return prev_origin if prev_origin is not None else game.ball.p.to_2d()


def carry_exhausted(game: Game, origin: Optional[Vector2D], stops_after: bool = False) -> bool:
    """True once the ball reaches `CARRY_LIMIT_M` from `origin` (see `carry_origin`):
    carrying further risks an excessive-dribbling foul, so release the ball.

    `stops_after`: the carrier brakes to a halt with the ball still on the dribbler
    once this fires (DecoyOverload's lure), so count the stopping distance too --
    without it a ~1.4m/s lure braked another ~0.4m and fouled at 1.01m anyway. Leave
    False for a carrier that kicks while moving (GiveAndGo): counting it there ended
    carries ~0.5m early and cost give_and_go_solo half its goals in an A/B
    (tournament_20260924_092119 bisect)."""
    if origin is None or game.ball is None:
        return False
    speed = math.hypot(game.ball.v.x, game.ball.v.y) if stops_after else 0.0
    stopping = speed**2 / (2 * _CARRY_BRAKE_DECEL_MPS2)
    return game.ball.p.to_2d().distance_to(origin) + stopping >= CARRY_LIMIT_M


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


_PASS_ROLLING_MPS = 0.5


def ball_line_receive_point(game: Game, receiver_id: int) -> Optional[Vector2D]:
    """Where `receiver_id` meets a ball already rolling towards it: its own position
    projected onto the ball's actual path. None while the ball is slower than
    `_PASS_ROLLING_MPS` or moving away from the receiver."""
    ball = game.ball
    velocity = Vector2D(ball.v.x, ball.v.y)
    speed = velocity.mag()
    if speed < _PASS_ROLLING_MPS:
        return None
    ball_pos = ball.p.to_2d()
    along = (game.friendly_robots[receiver_id].p - ball_pos).dot(velocity) / speed
    if along <= 0.0:
        return None
    return ball_pos + velocity * (along / speed)


def enemy_positions(game: Game) -> list[Vector2D]:
    return [enemy.p for enemy in game.enemy_robots.values() if enemy is not None]


_LOOSE_BALL_SPEED = 0.3  # m/s — matches goalkeep.py's own-box retrieval threshold
_LOOSE_BALL_CONTEST_RANGE = 1.5  # metres — matches PressAndContainTactic's own _PRESS_RANGE


def ball_is_loose(game: Game, contest_range: float = _LOOSE_BALL_CONTEST_RANGE) -> bool:
    """True when the ball is sitting dead with no enemy nearby able to contest it.

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

    An enemy within `contest_range` of the ball only counts as "contesting"
    it if the enemy is legally allowed to actually close that distance. If
    the ball is in/near our own defense area (only our keeper may enter it —
    `DefenseAreaRule` fouls any other robot, friend or enemy, that does), an
    enemy sitting just outside the boundary is already at its closest legal
    approach and can never advance further: it reads as "1m away" forever
    without ever being able to touch the ball. Found live 2026-09-02
    (`tiki_taka_vs_zone_fluid_Rk.pkl`, ticks 540-599): ball dead at
    `(-4.253, 1.030)`, ~0.03m outside `zone_fluid`'s own defense area (so
    `ball_in_own_defense_area` never sends the keeper either — see
    `goalkeep.py`'s `_ball_needs_retrieval`), with a `tiki_taka` attacker
    parked ~0.22m away at the boundary, barred from entering. Neither team's
    logic ever claimed the ball again for the rest of the match. Mirroring
    that enemy's position against `in_own_defense_area` catches exactly this
    "parked at the wall, can get no closer" case without needing to simulate
    an actual path.

    Deliberately does not check which team's corner the ball is in, or
    distance from any particular robot — that's for the caller (typically:
    "is my own nearest assigned robot closer to the ball than
    `_LOOSE_BALL_CONTEST_RANGE`, and if so send it to fetch the ball instead
    of holding its normal shape this tick").
    """
    if game.ball is None:
        return False
    # A teammate already dribbling the ball is the opposite of loose,
    # regardless of how slowly they're carrying it or whether any enemy is
    # nearby -- ball_speed alone can't tell "abandoned, rolling to a stop"
    # apart from "being carefully carried while lining up a pass" (both are
    # slow), and this function's own docstring only ever reasoned about
    # enemy contest range, never about the caller's own team. Found live,
    # 2026-09-03: `ShadowAndMarkTactic`'s retriever picked itself while its
    # own teammate (a different tactic's carrier) was dribbling slowly to
    # aim a pass, and drove straight into it -- a same-team scrum, both
    # robots then reading has_ball=True while the ball itself went nowhere.
    if any(robot.has_ball for robot in game.friendly_robots.values()):
        return False
    ball_speed = (game.ball.v.x**2 + game.ball.v.y**2) ** 0.5
    if ball_speed >= _LOOSE_BALL_SPEED:
        return False
    ball_pos = game.ball.p.to_2d()
    # A small margin beyond the bare legal rectangle: a ball resting just
    # outside the line (as in the live-found bug — 0.03m out) is still, in
    # practice, only reachable by whichever robot is allowed inside the box,
    # since any outfield robot approaching it must cross the boundary to
    # actually touch it. Matches the standard "just outside the box" margin
    # used elsewhere for this same reasoning (`clamp_outside_own_defense_area`,
    # `own_defense_area_exit_point`).
    ball_barred_for_enemies = in_own_defense_area(game, ball_pos, margin=2.0 * ROBOT_RADIUS + 0.05)
    for enemy in game.enemy_robots.values():
        if enemy is None:
            continue
        if enemy.p.distance_to(ball_pos) > contest_range:
            continue
        if ball_barred_for_enemies and not in_own_defense_area(game, enemy.p):
            # Ball is in our own defense area (only our keeper may enter —
            # see `in_own_defense_area`'s docstring); this enemy is outside
            # it and so cannot legally get any closer. Not a contester.
            continue
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


# Every `find_best_shot`/`_score_goal` call site picks a target y within
# [goal_y1, goal_y2] and gates the kick on `oriented_towards`'s fixed
# `ORIENTATION_TOLERANCE_RAD` (0.05 rad) — a robot "close enough" to that
# target orientation still has real lateral aim slop when the kick actually
# fires (`kick()` has no target of its own; the ball launches along the
# robot's live orientation at that instant, see `oriented_towards`'s
# callers), roughly `distance * tan(0.05)` for a shot taken close to
# straight-on and growing further for a sharper approach angle. If
# `_find_best_shot` is allowed to pick a target right at the true post edge,
# a shot that passes the orientation-tolerance check can still fly past the
# post and out of bounds instead of through the goal. Confirmed live
# (tournament replay clear_press_plus_vs_high_press, t=47.8s, shot distance
# ~2.4m at a ~50 deg approach angle): a `GiveAndGoTactic` shooter's kick
# landed within 0.45 deg of its own `target_oren` (the aim itself was
# accurate) but that target was ~1.7 deg outside the true goal-bottom edge,
# and the ball flew straight out via the boundary next to the goal rather
# than through it. Inset both posts by a fixed safety margin, calibrated
# against that traced geometry plus headroom, so a shot in that same
# realistic mid-range-and-angle envelope stays inside the real goal even at
# the tolerance boundary. This is a practical mitigation for the shot
# geometries this codebase's tactics actually produce, not a mathematical
# guarantee for every distance/angle combination — an extreme close-range,
# highly oblique shot (well under 1m from the line, aimed near-parallel to
# it) can still exceed this margin, since lateral slop from a fixed angular
# tolerance is unbounded as approach angle steepens. That case would need a
# distance/angle-aware correction per call site, not a fixed inset; not
# pursued here since it wasn't the mechanism observed live.
_GOAL_POST_SAFETY_MARGIN = 0.2  # metres — covers the traced ~2.4m/~50deg case with headroom


def enemy_goal_line(game: Game) -> tuple[float, float, float]:
    goal_line = game.field.enemy_goal_line
    goal_x = float(goal_line[0][0])
    raw_y1 = min(float(goal_line[0][1]), float(goal_line[1][1]))
    raw_y2 = max(float(goal_line[0][1]), float(goal_line[1][1]))
    # Degenerate/very narrow goals (e.g. a test double field) would invert
    # under a naive inset — clamp to the midpoint instead of crossing over.
    mid = (raw_y1 + raw_y2) / 2.0
    goal_y1 = min(raw_y1 + _GOAL_POST_SAFETY_MARGIN, mid)
    goal_y2 = max(raw_y2 - _GOAL_POST_SAFETY_MARGIN, mid)
    return goal_x, goal_y1, goal_y2


def in_own_defense_area(game: Game, point: Vector2D, margin: float = 0.0) -> bool:
    """True if `point` is inside our own defense area, optionally inflated by `margin`.

    Uses `field.my_defense_area` — the same geometry the CustomReferee's
    `DefenseAreaRule` derives from (`half_defense_area_depth`/`width`), so a
    tactic deciding legality by this check (at the default `margin=0.0`)
    agrees with the referee. `margin` widens the rectangle outward on both
    the front edge and the sides — for callers reasoning about "close enough
    to the box that an enemy parked at its edge can get no closer" rather
    than the bare legal boundary itself (see `ball_is_loose`).
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    goal_x = game.field.my_goal_line[0][0]
    half_width = abs(float(defense_area[0][1]))
    sign = 1.0 if (goal_x - front_x) >= 0.0 else -1.0
    x_inside = sign * (point.x - front_x) >= -margin
    return x_inside and abs(point.y) <= half_width + margin


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


def _beside_defense_area(game: Game, point: Vector2D, margin: float) -> bool:
    """True when `point` is wider than the box by more than `margin`: already legal and
    clear of the planner's ring, so the x clamp below must not pull it off its line
    (a receive point at (-3.41, 2.7) was dragged to x = -2.95, high_line_zone_vs_low_block
    2026-09-28)."""
    return abs(point.y) > game.field.half_defense_area_width + margin


def clamp_outside_own_defense_area(game: Game, point: Vector2D, margin: float = _DEFENSE_AREA_CLAMP_MARGIN) -> Vector2D:
    """Clamp a target point to just outside our own defense area's front edge.

    The `DefenseAreaRule` fouls any outfield robot entering the area (the
    keeper owns the box), so tactics that route robots near their own goal —
    shot-line defenders, carriers chasing a loose ball — must never target
    inside it. Keep-out is enforced on x with one robot-diameter margin;
    the y-coordinate is preserved unchanged (the area only spans
    `half_defense_area_width`, so a clamped x alone is already outside).
    """
    defense_area = game.field.my_defense_area
    if _beside_defense_area(game, point, margin):
        return point
    front_x = float(defense_area[1][0])
    sign = 1.0 if game.my_team_is_right else -1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def own_defense_area_exit_point(game: Game, at_y: float, margin: float = _DEFENSE_AREA_CLAMP_MARGIN) -> Vector2D:
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
    game: Game, point: Vector2D, margin: float = _DEFENSE_AREA_CLAMP_MARGIN
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
    if _beside_defense_area(game, point, margin):
        return point
    front_x = float(defense_area[1][0])
    sign = -1.0 if game.my_team_is_right else 1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def enemy_defense_area_hold_point(game: Game, at_y: float, margin: float = _DEFENSE_AREA_CLAMP_MARGIN) -> Vector2D:
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

    # Clearance against OTHER teammates only -- a robot standing at (or very
    # near) `passer_position`/`receiver_position` itself is presumably the
    # passer/receiver being scored, not a third robot in the way, and must
    # be excluded rather than counted as a clearance violation. Every real
    # caller passes a LIVE robot's own `.p` as one or both positions
    # (`_best_receiver` passes `game.friendly_robots[carrier_id].p` and
    # `game.friendly_robots[candidate_id].p` directly; `_nearest_safe_
    # receiver`'s sibling callers do the same) -- `game.friendly_robots`
    # still contains that exact same robot, so `distance_to` on the
    # unfiltered roster was always exactly 0.0 for at least the passer
    # itself, `< min_robot_clearance` every single time. Found live,
    # 2026-09-03: this meant `score_pass_setup` returned None on literally
    # every call from `_best_receiver`, which made that whole "pick the
    # highest-scoring teammate" path pure dead code -- every hop past the
    # first touch of a possession fell through to `_nearest_safe_receiver`
    # (deliberately unscored, nearest-with-clear-lane only) or a forced
    # shot, which is the actual reason passes looked short and low-value
    # regardless of any scoring-formula weighting (see `progress` above):
    # the formula was never being consulted at all.
    for robot_pos in game.friendly_robots.values():
        pos = robot_pos.p
        if (
            pos.distance_to(passer_position) < min_pass_distance
            or pos.distance_to(receiver_position) < min_pass_distance
        ):
            continue  # this is the passer or receiver itself, not a third robot
        if passer_position.distance_to(pos) < min_robot_clearance:
            return None
        if receiver_position.distance_to(pos) < min_robot_clearance:
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
    # Metres of net progress toward the enemy goal line this pass buys,
    # sign-corrected by which side `goal_x` is on so it's positive whenever
    # the receiver ends up closer to goal than the passer, negative for a
    # backward pass, ~0 for a square ball -- distinct from
    # `distance_to_goal_ratio` above, which only ever looks at the
    # receiver's absolute proximity to goal, not what THIS pass changed.
    # Added because the old formula had no such term at all: `shot_gap`/
    # `pass_clearance` (both O(1-3m), field-geometry-scale) completely
    # dominated the old `-0.03 * pass_distance` penalty (an 8m switch cost
    # only 0.24, less than a single extra metre of clearance), so a short
    # pass to whichever teammate happened to be standing nearby scored
    # about the same as a longer one that actually advanced the ball --
    # confirmed live, 2026-09-03: user-observed matches were full of short,
    # low-value passes under real risk/time cost with nothing to show for
    # them. Weighted at the same O(1) scale as shot_gap/pass_clearance
    # (unlike the old distance penalty) so a pass now has to buy either a
    # genuinely better shot or real territory to beat a shorter/safer one.
    progress = (receiver_position.x - passer_position.x) * (1.0 if goal_x > 0 else -1.0)
    score = shot_gap + 0.3 * pass_clearance + 0.5 * progress - 0.2 * distance_to_goal_ratio - 0.03 * pass_distance

    return PassSetupScore(passer_position=passer_position, receiver_position=receiver_position, score=score)
