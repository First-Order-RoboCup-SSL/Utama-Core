from typing import Optional

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.run.predictors.position import predict_ball_pos_at_x
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import face_ball, move


def defend_parameter(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    defenseing_friendly = game.friendly_robots[robot_id]
    vel = game.ball.v.to_2d()
    if vel[0]**2 + vel[1]**2 > 0.05:
        x1, y1 = game.ball.p.x, game.ball.p.y
        x2, y2 = 4.5, 0.5

        x3, y3 = defenseing_friendly.p.x, defenseing_friendly.p.y

        t = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
        x4 = x1 + t * (x2 - x1)
        y4 = y1 + t * (y2 - y1)

        target_pos = np.array([x4, y4])
        return go_to_point(
                game,
                motion_controller,
                robot_id,
                Vector2D(target_pos[0], target_pos[1]),
                dribbling=True,
            )
        
    shooting_enemy = game.enemy_robots[0]

    robot_rad = 0.09 

    x1, y1 = game.ball.p.x, game.ball.p.y
    x2, y2 = 4.5, -0.5 

    x3, y3 = defenseing_friendly.p.x, defenseing_friendly.p.y

    t = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
    x4 = x1 + t * (x2 - x1)
    y4 = y1 + t * (y2 - y1)

    vec_to_target = np.array([x4 - x3, y4 - y3])
    dist_to_target = np.linalg.norm(vec_to_target)

    if dist_to_target > 0:
        vec_dir = vec_to_target / dist_to_target
    else:
        vec_dir = np.array([0.0, 0.0])

    target_pos = np.array([x4, y4]) - vec_dir * robot_rad

    target_oren = np.pi if game.my_team_is_right else 0
    return move(
        game,
        motion_controller,
        robot_id,
        Vector2D(target_pos[0], target_pos[1]),
        target_oren,
        True
    )