from typing import Optional

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.run.predictors.position import predict_ball_pos_at_x
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import face_ball, move


def mykeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    shooting_enemy = game.enemy_robots[robot_id]
    defenseing_friendly = game.friendly_robots[robot_id]
    if shooting_enemy.has_ball:
        robot_rad = 0.09  # radius of robot in meters
        # Calculate the perpendicular projection point from the robot to the goal line
        x1 = shooting_enemy.x
        y1 = shooting_enemy.y
        x2 = 4.5
        y2 = -0.5
        x3 = defenseing_friendly.x
        y3 = defenseing_friendly.y
        t = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
        t /= (x2 - x1) ** 2 + (y2 - y1) ** 2
        x4 = x1 + t * (x2 - x1)
        y4 = y1 + t * (y2 - y1)
        target_pos = Vector2D(x4, y4) - Vector2D(x4 - x3, y4 - y3).normalized() * robot_rad
        target_oren = Vector2D(x3 - x1, y3 - y1).angle()

        target_oren = np.pi if game.my_team_is_right else 0
        return move(
            game,
            motion_controller,
            robot_id,
            target_pos,
            target_oren,
            True,
        )

    if game.my_team_is_right:
        target = predict_ball_pos_at_x(game, 4.5)
    else:
        target = predict_ball_pos_at_x(game, -4.5)

    if not target or abs(target[1]) > 0.5:
        target = Vector2D(4.5 if game.my_team_is_right else -4.5, 0)

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
