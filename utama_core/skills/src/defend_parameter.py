import math
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

    def positions_to_defend_parameter(x2, y2):
        x1, y1 = game.ball.p.x, game.ball.p.y
        x3, y3 = defenseing_friendly.p.x, defenseing_friendly.p.y
        t = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x4 = x1 + t * (x2 - x1)
        y4 = y1 + t * (y2 - y1)

        def cal_xy5(xa, ya, xb, yb, w, x4, y4):
            x5 = xa - w
            dx = xb - xa
            if abs(dx) < 1e-12:
                return x4, y4
            t = (x5 - xa) / dx
            y5 = ya + t * (yb - ya)
            return x5, y5

        def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if abs(x2 - x4) < 1 or abs(y2 - y4) < 2:
            hori_x, hori_y = cal_xy5(x2, y2, x1, y1, 1.0, x4, y4)
            ver_x, ver_y = cal_xy5(y2, x2, y1, x1, 2.0, x4, y4)
            if distance(x4, y4, hori_x, hori_y) < distance(x4, y4, ver_x, ver_y):
                x4, y4 = hori_x, hori_y
            else:
                x4, y4 = ver_y, ver_x

        return x3, y3, x4, y4

    # def modified_pos(target_pos):
    #     if target_pos[0] > 4.0:
    #         if target_pos[1] < 1.0 and target_pos[1] >= 0:
    #             if 4.5 - target_pos[0] < target_pos[1] - 1.0:
    #                 target_pos[0] = 4.0
    #             else:
    #                 target_pos[1] = 1.0
    #         elif target_pos[1] > -1.0 and target_pos[1] < 0:
    #             if 4.5 - target_pos[0] < -1.0 - target_pos[1]:
    #                 target_pos[0] = 4.0
    #             else:
    #                 target_pos[1] = -1.0
    #     return target_pos

    if vel[0] ** 2 + vel[1] ** 2 > 0.05:
        x2, y2 = 4.5, 0.5
        # target_pos = modified_pos(np.array([positions_to_defend_parameter(x2, y2)[2], positions_to_defend_parameter(x2, y2)[3]]))
        target_pos = np.array([positions_to_defend_parameter(x2, y2)[2], positions_to_defend_parameter(x2, y2)[3]])

    else:
        robot_rad = 0.09
        x2, y2 = 4.5, -0.5
        vec_to_target = np.array(
            [
                positions_to_defend_parameter(x2, y2)[2] - positions_to_defend_parameter(x2, y2)[0],
                positions_to_defend_parameter(x2, y2)[3] - positions_to_defend_parameter(x2, y2)[1],
            ]
        )
        dist_to_target = np.linalg.norm(vec_to_target)

        if dist_to_target > 0:
            vec_dir = vec_to_target / dist_to_target
        else:
            vec_dir = np.array([0.0, 0.0])
        # target_pos = modified_pos(np.array([positions_to_defend_parameter(x2, y2)[2], positions_to_defend_parameter(x2, y2)[3]]) - vec_dir * robot_rad)
        target_pos = (
            np.array([positions_to_defend_parameter(x2, y2)[2], positions_to_defend_parameter(x2, y2)[3]])
            - vec_dir * robot_rad
        )

    return go_to_point(game, motion_controller, robot_id, Vector2D(target_pos[0], target_pos[1]), dribbling=True)
