from typing import Optional

import numpy as np

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import face_ball, move

# TODO: instead of checking number of friendly, should check roles


def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    EDGE_OFFSET = BALL_RADIUS + ROBOT_RADIUS
    goal_x = game.field.my_goal_line[0][0]
    half_goal_width = game.field.half_goal_width
    target = predict_ball_pos_at_x(game, goal_x)

    stop_y = 0.0

    def intersection_with_vertical_line(a, b):
        xa, ya = a
        xb, yb = b

        if xb == xa:
            return a  # Line is vertical, return the point itself

        k = (yb - ya) / (xb - xa)
        y_intersect = ya + k * (goal_x - xa)
        if y_intersect < -half_goal_width:
            return (goal_x, -half_goal_width)
        elif y_intersect > half_goal_width:
            return (goal_x, half_goal_width)
        return (goal_x, y_intersect)

    if len(game.friendly_robots) == 2:
        try:
            # Check if defender is between ball and goal (side-aware)
            defender_between = (game.my_team_is_right and game.friendly_robots[1].p.x > game.ball.p.x) or (
                not game.my_team_is_right and game.friendly_robots[1].p.x < game.ball.p.x
            )
            if defender_between:
                # 1. Project BOTH edges to find the defender's shadow on the goal line
                _, yy_top = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y),
                    (
                        game.friendly_robots[1].p.x,
                        game.friendly_robots[1].p.y + EDGE_OFFSET,
                    ),
                )
                _, yy_bottom = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y),
                    (
                        game.friendly_robots[1].p.x,
                        game.friendly_robots[1].p.y - EDGE_OFFSET,
                    ),
                )

                # 2. Calculate the size of the gaps (using max(0, ...) to ignore negative space if shadow is outside the goal)
                top_gap_size = max(0, half_goal_width - yy_top)
                bottom_gap_size = max(0, yy_bottom - (-half_goal_width))

                # 3. Position the goalie in the middle of the LARGEST gap
                if top_gap_size > bottom_gap_size:
                    stop_y = (yy_top + half_goal_width) / 2
                else:
                    stop_y = (yy_bottom - half_goal_width) / 2
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
                    (game.ball.p.x, game.ball.p.y),
                    (
                        game.friendly_robots[1].p.x,
                        game.friendly_robots[1].p.y + EDGE_OFFSET,
                    ),
                )
                _, yy2 = intersection_with_vertical_line(
                    (game.ball.p.x, game.ball.p.y),
                    (
                        game.friendly_robots[2].p.x,
                        game.friendly_robots[2].p.y - EDGE_OFFSET,
                    ),
                )
                stop_y = (yy1 + yy2) / 2
        except (IndexError, KeyError):
            # If robots with IDs 1 or 2 are not available, keep existing stop_y
            pass
    if not target or abs(target[1]) > half_goal_width:
        target = Vector2D(goal_x, stop_y)

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
