from typing import Optional

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.defense_utils import (
    clamp_y,
    intersection_with_x_line,
    single_defender_stop_y,
)


def _defender_target_y(ball_pos, keeper_x, goal_frame_y, edge_offset, defender_x, defense_half_width):
    """Compute the defender y-position for a given goal_frame_y assignment."""
    offset = edge_offset if goal_frame_y >= 0 else -edge_offset
    goal_point_y = goal_frame_y + offset

    dx = keeper_x - ball_pos.x
    if abs(dx) > 1e-12:
        t = (defender_x - ball_pos.x) / dx
        t = max(0.0, min(1.0, t))
        target_y = ball_pos.y + t * (goal_point_y - ball_pos.y)
    else:
        target_y = goal_frame_y

    return max(-defense_half_width, min(defense_half_width, target_y))


def _shadow_width(ball_pos, defender_pos, keeper_x, post_limit, edge_offset):
    """Width of the shadow interval a defender casts on the keeper line."""
    _, yy_top = intersection_with_x_line(
        (ball_pos.x, ball_pos.y),
        (defender_pos.x, defender_pos.y + edge_offset),
        keeper_x,
        post_limit,
    )
    _, yy_bottom = intersection_with_x_line(
        (ball_pos.x, ball_pos.y),
        (defender_pos.x, defender_pos.y - edge_offset),
        keeper_x,
        post_limit,
    )
    return abs(yy_top - yy_bottom)


def _choose_defender_side(
    game,
    ball_pos,
    keeper_x,
    post_limit,
    edge_offset,
    defender_x,
    defense_half_width,
    robot_id,
):
    """For a single defender (2-robot team), dynamically pick the better post side."""
    # Keeper reference: predicted intercept > current keeper y > centre
    keeper_ref = 0.0
    try:
        predicted = predict_ball_pos_at_x(game, keeper_x)
        if predicted is not None:
            keeper_ref = clamp_y(predicted.y, post_limit)
        else:
            keeper_ref = clamp_y(game.friendly_robots[0].p.y, post_limit)
    except (IndexError, KeyError, AttributeError):
        pass

    candidates = [post_limit, -post_limit]
    best_side = post_limit if robot_id == 1 else -post_limit
    best_score = None

    for side in candidates:
        target_y = _defender_target_y(
            ball_pos,
            keeper_x,
            side,
            edge_offset,
            defender_x,
            defense_half_width,
        )
        defender_pos = Vector2D(defender_x, target_y)
        stop_y = single_defender_stop_y(
            ball_pos,
            defender_pos,
            keeper_x,
            post_limit,
            edge_offset,
        )
        dist = abs(stop_y - keeper_ref)
        shadow_w = _shadow_width(
            ball_pos,
            defender_pos,
            keeper_x,
            post_limit,
            edge_offset,
        )
        # Lower is better: closest keeper stop_y, then largest shadow, then id convention
        id_tiebreak = 0 if (robot_id == 1) == (side > 0) else 1
        score = (dist, -shadow_w, id_tiebreak)

        if best_score is None or score < best_score:
            best_score = score
            best_side = side

    return best_side


def defend_parameter(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
    goal_frame_y: Optional[float] = None,
):
    ball_pos = game.ball.p.to_2d()

    goal_x = game.field.my_goal_line[0][0]
    goal_half_width = game.field.half_goal_width
    defense_area = game.field.my_defense_area
    defense_front_x = float(defense_area[1][0])
    defender_x = defense_front_x + (ROBOT_RADIUS if not game.my_team_is_right else -ROBOT_RADIUS)
    defense_half_width = abs(float(defense_area[0][1]))

    keeper_x = goal_x + (ROBOT_RADIUS if not game.my_team_is_right else -ROBOT_RADIUS)
    post_limit = goal_half_width - ROBOT_RADIUS
    edge_offset = BALL_RADIUS + ROBOT_RADIUS

    if goal_frame_y is None:
        if len(game.friendly_robots) == 2:
            goal_frame_y = _choose_defender_side(
                game,
                ball_pos,
                keeper_x,
                post_limit,
                edge_offset,
                defender_x,
                defense_half_width,
                robot_id,
            )
        else:
            goal_frame_y = post_limit if robot_id == 1 else -post_limit

    target_y = _defender_target_y(
        ball_pos,
        keeper_x,
        goal_frame_y,
        edge_offset,
        defender_x,
        defense_half_width,
    )
    target = Vector2D(defender_x, target_y)

    return go_to_point(game, motion_controller, robot_id, target, dribbling=True)
