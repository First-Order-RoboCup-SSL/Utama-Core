"""`goalkeep` — position the goalkeeper on the goal line to cover the predicted shot.

Called directly on robot 0 (pinned outside the kernel scheduler — see
`docs/STRATEGY_DEVELOPMENT.md`'s "Single-writer partition invariant"), not
wrapped in a `Tactic`. The keeper's target y-position on the goal line is
`predict_ball_pos_at_x`'s intercept when available, adjusted to account for
one or two outfield defenders standing between the ball and goal (so the
keeper doesn't try to cover an angle a teammate is already shadowing) — see
the 1/2/3+ friendly-robot branches below.

Exception: a ball at rest inside our own defense area never gets a goal-line
target at all — see `_ball_needs_retrieval`'s docstring. Every outfield
tactic that reaches a ball there holds the box's front edge instead of
entering (`DefenseAreaRule` fouls any non-keeper inside), and this was the
one function that could legally go in and do something about it, but never
did: it always drove to a goal-line intercept regardless of where the ball
actually was, so a ball that rolled dead in the box (not heading at goal,
`predict_ball_pos_at_x` returns `None` for a near-stationary ball) just sat
there for the rest of the match. Found live: a `clear_danger` vs `low_block`
match pinned for the last 28 s of a 60 s game this way (see
`docs/strategies.md`'s "Known open bugs").
"""

from typing import Optional

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    has_ball,
    oriented_towards,
    own_defense_area_exit_point,
)
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.defense_utils import (
    clamp_y,
    intersection_with_x_line,
    single_defender_stop_y,
)
from utama_core.skills.src.utils.move_utils import kick, move, turn_on_spot

# TODO: instead of checking number of friendly, should check roles

_RETRIEVE_BALL_SPEED = 0.3  # m/s — below this, a ball in our box is "at rest," not a live shot to block
_CLEAR_ARRIVED_MARGIN = ROBOT_RADIUS + 0.1  # how close to the box exit point counts as "arrived, ready to kick"

# `predict_ball_pos_at_x` returns None the instant the ball's x reaches (or
# passes) `keeper_x`, since `t = (x - pos.x) / vel.x` goes negative right at
# that crossing -- including the single tick a live shot actually crosses
# the goal line, which is exactly the moment a stable target matters most.
# Within this x-distance of the line, the ball's raw `ball_pos.y` is itself
# already an accurate stand-in for the vanished prediction (the ball is
# right there), so use that instead of falling back to `stop_y` -- the
# goal's center by default, often the opposite direction from an incoming
# shot. See the `target is None` branch in `goalkeep` below.
_NEAR_LINE_DISTANCE = 0.5  # m


def _ball_needs_retrieval(game: Game, robot_id: int) -> bool:
    """A ball at rest inside our own box is not a shot to block — it is a
    ball only the keeper may legally go and clear (`DefenseAreaRule` caps
    outfield entry at 0). `goalkeep`'s ordinary goal-line targeting has no
    concept of this: a stationary ball gives `predict_ball_pos_at_x` nothing
    to predict (returns `None`), so the keeper falls back to a `stop_y`
    computed from the ball's *current* y, which is still a goal-line point,
    not the ball's actual position — so it never converges on a ball that
    isn't already on the line. Once the keeper has picked the ball up,
    `_ball_needs_clearing` takes over (see that docstring) instead of this
    going back to False and falling through to the ordinary goal-line
    branch, which would dribble the retrieved ball right back toward goal.
    """
    if game.ball is None or not ball_in_own_defense_area(game):
        return False
    if has_ball(game, robot_id, visual=True):
        return False
    ball_speed = (game.ball.v.x**2 + game.ball.v.y**2) ** 0.5
    return ball_speed < _RETRIEVE_BALL_SPEED


def _ball_needs_clearing(game: Game, robot_id: int) -> bool:
    """True whenever the keeper currently has the ball — the follow-up to
    `_ball_needs_retrieval`, covering the whole dribble-to-exit-then-kick
    sequence. Deliberately does NOT also require `ball_in_own_defense_area`:
    the ball tracks the dribbling keeper, so by the time it reaches the box's
    exit point (the dribble target in `goalkeep` below) it has already
    crossed to just outside the box — gating on box position would go False
    right at arrival and drop back to the ordinary goal-line branch
    mid-clearance, dribbling the retrieved ball right back in. `has_ball`
    alone is the right latch: as soon as the keeper actually kicks (the last
    step of this branch), it no longer has the ball and this goes False on
    its own, with nothing left to clear.

    Note this means an ordinary save where the keeper ends up holding the
    ball right on the goal line also routes here rather than the goal-line
    branch — harmless: the exit point sits further from goal than the line,
    so the keeper simply dribbles it there and clears, which is a reasonable
    thing to do with a held ball regardless of how it got there.
    """
    if game.ball is None:
        return False
    return has_ball(game, robot_id, visual=True)


def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
):
    """Move `robot_id` (the goalkeeper) to cover the predicted shot on the goal line.

    Returns `None` if `game.ball` is unset (nothing to react to); otherwise a
    `RobotCommand` moving the keeper to the goal line, with the target
    y-coordinate adjusted for however many outfield defenders (0, 1, or 2+)
    are currently positioned between the ball and the goal. Exception: a ball
    at rest in our own box is driven straight to (see `_ball_needs_retrieval`)
    rather than treated as a shot to cover.
    """
    if game.ball is None:
        return None

    if _ball_needs_retrieval(game, robot_id):
        return go_to_point(
            game,
            motion_controller,
            robot_id,
            game.ball.p.to_2d(),
            dribbling=True,
        )

    if _ball_needs_clearing(game, robot_id):
        # Dribble to the box's front edge, then kick square upfield (away
        # from our own goal, along the field's long axis) — a minimal
        # clearance. Not `ClearBallTactic`'s lane-scored clearance: that
        # tactic is an outfield-robot slot with room to evaluate candidate
        # landing lanes; the keeper's only job here is "don't leave the ball
        # sitting dead in the box," so the simplest kick that gets it out
        # and moving is enough.
        keeper = game.friendly_robots[robot_id]
        exit_point = own_defense_area_exit_point(game, keeper.p.y)
        upfield_sign = -1.0 if game.my_team_is_right else 1.0
        clear_target = Vector2D(exit_point.x + upfield_sign * 2.0, exit_point.y)
        target_oren = keeper.p.angle_to(clear_target)
        if keeper.p.distance_to(exit_point) >= _CLEAR_ARRIVED_MARGIN:
            return move(
                game=game,
                motion_controller=motion_controller,
                robot_id=robot_id,
                target_coords=exit_point,
                target_oren=target_oren,
                dribbling=True,
            )
        if oriented_towards(game, robot_id, target_oren):
            return kick()
        return turn_on_spot(
            game=game,
            motion_controller=motion_controller,
            robot_id=robot_id,
            target_oren=target_oren,
            dribbling=True,
        )

    edge_offset = BALL_RADIUS + ROBOT_RADIUS
    goal_x = game.field.my_goal_line[0][0]
    keeper_x = goal_x + (ROBOT_RADIUS if not game.my_team_is_right else -ROBOT_RADIUS)
    goal_half_width = game.field.half_goal_width
    post_limit = goal_half_width - ROBOT_RADIUS
    ball_pos = game.ball.p.to_2d()
    target = predict_ball_pos_at_x(game, keeper_x)

    stop_y = 0.0

    if len(game.friendly_robots) == 1:
        stop_y = clamp_y(ball_pos.y, post_limit)
    elif len(game.friendly_robots) == 2:
        try:
            # Check if defender is between ball and goal (side-aware)
            defender_between = (game.my_team_is_right and game.friendly_robots[1].p.x > ball_pos.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < ball_pos.x
            )
            if defender_between:
                stop_y = single_defender_stop_y(
                    ball_pos,
                    game.friendly_robots[1].p,
                    keeper_x,
                    post_limit,
                    edge_offset,
                )
        except (IndexError, KeyError):
            # If robot with ID 1 is not available, keep default stop_y
            pass
    elif len(game.friendly_robots) >= 3:
        try:
            # Check if both defenders are between ball and goal (side-aware)
            defender1_between = (game.my_team_is_right and game.friendly_robots[1].p.x > ball_pos.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < ball_pos.x
            )
            defender2_between = (game.my_team_is_right and game.friendly_robots[2].p.x > ball_pos.x) or (
                not game.my_team_is_right and game.friendly_robots[2].p.x < ball_pos.x
            )
            if defender1_between and defender2_between:
                _, yy1 = intersection_with_x_line(
                    (ball_pos.x, ball_pos.y),
                    (game.friendly_robots[1].p.x, game.friendly_robots[1].p.y + edge_offset),
                    keeper_x,
                    post_limit,
                )
                _, yy2 = intersection_with_x_line(
                    (ball_pos.x, ball_pos.y),
                    (game.friendly_robots[2].p.x, game.friendly_robots[2].p.y - edge_offset),
                    keeper_x,
                    post_limit,
                )
                stop_y = (yy1 + yy2) / 2
        except (IndexError, KeyError):
            # If robots with IDs 1 or 2 are not available, keep existing stop_y
            pass
    if target is None:
        if abs(ball_pos.x - keeper_x) < _NEAR_LINE_DISTANCE:
            # The ball is right at the line -- most likely `predict_ball_pos_
            # at_x` just lost the shot because `t` crossed zero, not because
            # there's genuinely nothing to cover. Track the ball's own
            # position (clamped to the posts) instead of snapping away to
            # `stop_y`, which is often the opposite side of the goal from an
            # incoming shot and would otherwise yank the keeper off a save
            # in progress on exactly this tick.
            clamped_y = max(-post_limit, min(post_limit, ball_pos.y))
            target = Vector2D(keeper_x, clamped_y)
        else:
            target = Vector2D(keeper_x, stop_y)
    elif abs(target.y) > goal_half_width:
        # Ball heading toward goal but predicted wide -- clamp to nearest post
        # instead of snapping to stop_y, so the keeper stays reactive to the shot
        clamped_y = max(-post_limit, min(post_limit, target.y))
        target = Vector2D(keeper_x, clamped_y)
    else:
        target = Vector2D(keeper_x, target.y)

    return go_to_point(
        game,
        motion_controller,
        robot_id,
        target,
        dribbling=True,
    )
