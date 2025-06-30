from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from motion_planning.src.motion_controller import MotionController
from typing import Optional, Tuple
from skills.src.go_to_point import go_to_point
from skills.src.utils.defense_utils import find_likely_enemy_shooter
from run.predictors.position import predict_ball_pos_at_x
from entities.data.vector import Vector2D

import numpy as np

def goalkeep(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
):
    goalie_obj = game.friendly_robots[robot_id]
    if goalie_obj.has_ball:
        target_oren = np.pi if game.my_team_is_right else 0
        return go_to_point(
            game,
            motion_controller,
            goalie_obj.p,
            robot_id,
            ((4 if game.my_team_is_right else -4), 0),
            target_oren,
            True,
        )

    if game.my_team_is_right:
        target = predict_ball_pos_at_x(game, 4.5)
    else:
        target = predict_ball_pos_at_x(game, -4.5)

    if not target or abs(target[1]) > 0.5:
        target = Vector2D(4.5 if game.my_team_is_right else -4.5, 0)

    if target and not find_likely_enemy_shooter(
        game.enemy_robots, game.ball
    ):
        cmd = go_to_point(
            game,
            motion_controller,
            robot_id,
            target,
            dribbling=True,
        )
    return cmd
