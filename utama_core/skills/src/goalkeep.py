"""`goalkeep` — position the goalkeeper on the goal line to cover the predicted shot.

Called directly on robot 0 (pinned outside the kernel scheduler — see
`docs/STRATEGY_DEVELOPMENT.md`'s "Single-writer partition invariant"), not
wrapped in a `Tactic`. The keeper's target y-position on the goal line is
`predict_ball_pos_at_x`'s intercept when available, adjusted to account for
one or two outfield defenders standing between the ball and goal (so the
keeper doesn't try to cover an angle a teammate is already shadowing) — see
the 1/2/3+ friendly-robot branches below.
"""

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.defense_utils import (
    clamp_y,
    intersection_with_x_line,
    single_defender_stop_y,
)

# TODO: instead of checking number of friendly, should check roles


def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
):
    """Move `robot_id` (the goalkeeper) to cover the predicted shot on the goal line.

    Returns `None` if `game.ball` is unset (nothing to react to); otherwise a
    `RobotCommand` moving the keeper to the goal line, with the target
    y-coordinate adjusted for however many outfield defenders (0, 1, or 2+)
    are currently positioned between the ball and the goal.
    """
    if game.ball is None:
        return None

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
