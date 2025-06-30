import numpy as np
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from entities.data.command import RobotCommand
from motion_planning.src.motion_controller import MotionController
from typing import List, Optional, Tuple

from skills.src.go_to_point import go_to_point
from skills.src.utils.defense_utils import (
    align_defenders,
    find_likely_enemy_shooter,
    to_defense_parametric,
    velocity_to_orientation,
)
from skills.src.utils.move_utils import face_ball
from entities.data.vector import Vector2D

def defend_parameter(
    game: Game,
    motion_controller: MotionController,
    defender_id: int,
    env: Optional[SSLStandardEnv] = None,
) -> RobotCommand:
    friendly, enemy, ball = game.friendly_robots, game.enemy_robots, game.ball
    shooters_data = find_likely_enemy_shooter(enemy, ball)
    orientation = None
    tracking_ball = False

    if not shooters_data:
        target_tracking_coord = ball.p.to_2d()
        if (
            ball.v is not None
            and None not in ball.v
        ):
            orientation = velocity_to_orientation(ball.v)
            tracking_ball = True
    else:
        # TODO (deploy more defenders, or find closest shooter?)
        sd = shooters_data[0]
        target_tracking_coord = Vector2D(sd.x, sd.y)
        orientation = sd.orientation

    real_def_pos = game.friendly_robots[defender_id].p
    current_def_parametric = to_defense_parametric(game, real_def_pos)
    target = align_defenders(
        game,
        current_def_parametric,
        target_tracking_coord,
        orientation,
        env
    )
    cmd = go_to_point(
        game,
        motion_controller,
        defender_id,
        target,
        dribbling=True,
    )

    gp = (game.field.my_goal_line.coords[0][0], 0)
    if env:
        env.draw_line(
            [gp, (target_tracking_coord[0], target_tracking_coord[1])],
            width=5,
            color="RED" if tracking_ball else "PINK",
        )

    return cmd
