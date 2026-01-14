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
    shooting_enemy = game.enemy_robots[robot_id]
    defenseing_friendly = game.friendly_robots[robot_id]

    robot_rad = 0.09  # radius of robot in meters
    # Calculate the perpendicular projection point from the robot to the goal line
    x1 = shooting_enemy.p.x
    y1 = shooting_enemy.p.y
    x2 = 4.5
    y2 = -0.5
    x3 = defenseing_friendly.p.x
    y3 = defenseing_friendly.p.y
    t = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
    t /= (x2 - x1) ** 2 + (y2 - y1) ** 2
    x4 = x1 + t * (x2 - x1)
    y4 = y1 + t * (y2 - y1)

    def normalize_vector(v: Vector2D) -> Vector2D:
        norm = np.sqrt(v.x**2 + v.y**2)
        if norm == 0:
            return Vector2D(0, 0)
        return Vector2D(v.x / norm, v.y / norm)

    def vector_angle(v: Vector2D) -> float:
        return np.arctan2(v.y, v.x)

    target_pos = Vector2D(x4, y4) - normalize_vector(Vector2D(x4 - x3, y4 - y3)) * robot_rad
    target_oren = vector_angle(Vector2D(x3 - x1, y3 - y1))

    target_oren = np.pi if game.my_team_is_right else 0
    return move(
        game,
        motion_controller,
        robot_id,
        target_pos,
        target_oren,
        True,
    )
