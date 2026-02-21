from typing import Optional

import numpy as np

from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import face_ball, move


def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    if game.my_team_is_right:
        target = predict_ball_pos_at_x(game, 4.5)
    else:
        target = predict_ball_pos_at_x(game, -4.5)

    stop_y = 0.0

    def intersection_with_vertical_line(a, b, x_line=4.5):
        xa, ya = a
        xb, yb = b
        if xb < xa:
            return a

        k = (yb - ya) / (xb - xa)
        y_intersect = ya + k * (x_line - xa)
        if y_intersect < -0.5:
            return (x_line, -0.5)
        elif y_intersect > 0.5:
            return (x_line, 0.5)
        return (x_line, y_intersect)

    if len(game.friendly_robots) == 2:
        try:
            # Check if defender is between ball and goal (side-aware)
            defender_between = (game.my_team_is_right and game.friendly_robots[1].p.x > game.ball.p.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < game.ball.p.x
            )
            if defender_between:
                _, yy = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y), (game.friendly_robots[1].p.x, game.friendly_robots[1].p.y + 0.1)
                )
                stop_y = (yy + 0.5) / 2
        except (IndexError, KeyError):
            # If robot with ID 1 is not available, keep default stop_y
            pass
    elif len(game.friendly_robots) >= 3:
        try:
            # Check if both defenders are between ball and goal (side-aware)
            defender1_between = (game.my_team_is_right and game.friendly_robots[1].p.x > game.ball.p.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < game.ball.p.x
            )
            defender2_between = (game.my_team_is_right and game.friendly_robots[2].p.x > game.ball.p.x) or (
                not game.my_team_is_right and game.friendly_robots[2].p.x < game.ball.p.x
            )
            if defender1_between and defender2_between:
                _, yy1 = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y), (game.friendly_robots[1].p.x, game.friendly_robots[1].p.y + 0.1)
                )
                _, yy2 = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y), (game.friendly_robots[2].p.x, game.friendly_robots[2].p.y - 0.1)
                )
                stop_y = (yy1 + yy2) / 2
        except (IndexError, KeyError):
            # If robots with IDs 1 or 2 are not available, keep existing stop_y
            pass
    if not target or abs(target[1]) > 0.5:
        target = Vector2D(4.5 if game.my_team_is_right else -4.5, stop_y)

    # shooters_data = find_likely_enemy_shooter(game.enemy_robots, game.ball)

    if target:
        cmd = go_to_point(
            game,
            motion_controller,
            robot_id,
            target,
            dribbling=True,
        )
    else:
        cmd = move(
            game,
            motion_controller,
            robot_id,
            Vector2D(None, None),  # No specific target
            face_ball(game.friendly_robots[robot_id].p, game.ball.p),
        )
    return cmd
