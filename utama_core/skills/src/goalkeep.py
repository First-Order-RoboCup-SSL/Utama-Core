from typing import Optional

from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_point import go_to_point

SHADOW_EDGE_OFFSET = 0.1


def _clamp_goal_y(y: float, goal_half_width: float) -> float:
    return max(-goal_half_width, min(goal_half_width, y))


def _intersection_with_goal_line(a, b, goal_x: float, goal_half_width: float):
    xa, ya = a
    xb, yb = b
    dx = xb - xa

    if abs(dx) < 1e-12:
        return (goal_x, _clamp_goal_y(yb, goal_half_width))

    t = (goal_x - xa) / dx
    if t < 0:
        return (goal_x, _clamp_goal_y(ya, goal_half_width))

    y_intersect = ya + t * (yb - ya)
    return (goal_x, _clamp_goal_y(y_intersect, goal_half_width))


def _single_defender_stop_y(ball_pos: Vector2D, defender_pos: Vector2D, goal_x: float, goal_half_width: float) -> float:
    open_top = defender_pos.y <= ball_pos.y
    edge_y = defender_pos.y + SHADOW_EDGE_OFFSET if open_top else defender_pos.y - SHADOW_EDGE_OFFSET
    _, yy = _intersection_with_goal_line(
        (ball_pos.x, ball_pos.y),
        (defender_pos.x, edge_y),
        goal_x,
        goal_half_width,
    )

    if open_top:
        return (yy + goal_half_width) / 2

    return (yy - goal_half_width) / 2


def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    goal_line = game.field.my_goal_line
    goal_x = float(goal_line[0][0])
    goal_half_width = abs(float(goal_line[0][1]))
    ball_pos = game.ball.p.to_2d()
    target = predict_ball_pos_at_x(game, goal_x)

    stop_y = 0.0

    if len(game.friendly_robots) == 2:
        try:
            # Check if defender is between ball and goal (side-aware)
            defender_between = (game.my_team_is_right and game.friendly_robots[1].p.x > ball_pos.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < ball_pos.x
            )
            if defender_between:
                stop_y = _single_defender_stop_y(
                    ball_pos,
                    game.friendly_robots[1].p,
                    goal_x,
                    goal_half_width,
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
                _, yy1 = _intersection_with_goal_line(
                    (ball_pos.x, ball_pos.y),
                    (game.friendly_robots[1].p.x, game.friendly_robots[1].p.y + SHADOW_EDGE_OFFSET),
                    goal_x,
                    goal_half_width,
                )
                _, yy2 = _intersection_with_goal_line(
                    (ball_pos.x, ball_pos.y),
                    (game.friendly_robots[2].p.x, game.friendly_robots[2].p.y - SHADOW_EDGE_OFFSET),
                    goal_x,
                    goal_half_width,
                )
                stop_y = (yy1 + yy2) / 2
        except (IndexError, KeyError):
            # If robots with IDs 1 or 2 are not available, keep existing stop_y
            pass
    if target is None or abs(target.y) > goal_half_width:
        target = Vector2D(goal_x, stop_y)

    return go_to_point(
        game,
        motion_controller,
        robot_id,
        target,
        dribbling=True,
    )
