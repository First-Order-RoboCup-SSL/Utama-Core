import sys
import os
from typing import List
import numpy as np
import pytest
from entities.game.game_object import Colour, GameObject, Robot as GameRobot

from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from motion_planning.src.planning.controller import HybridWaypointMotionController
from robot_control.src.skills import (
    get_goal_centre,
    go_to_ball,
    go_to_point,
    align_defenders,
    mag,
    to_defense_parametric,
    face_ball,
    velocity_to_orientation,
)
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import find_likely_enemy_shooter, score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from robot_control.src.tests.utils import one_robot_placement, setup_pvp
from motion_planning.src.planning.path_planner import DynamicWindowPlanner
from robot_control.src.find_best_shot import ROBOT_RADIUS 
import random
import logging
import time
from math import dist

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)



def test_pathfinding(headless: bool, moving: bool):
    game = Game()
    N_ROBOTS_YELLOW = 6
    N_ROBOTS_BLUE = 6

    random.seed(0)
    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )

    mover_id = 1
    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)
    slow_pid2d = TwoDPID(TIMESTEP, 1, -1, 2, 0.1, 0.0, num_robots=6, normalize=False)

    env.reset()
    env.teleport_ball(2.25, -1)
    env.teleport_robot(True, 3, 0, -5, -1)
    env.teleport_robot(True, mover_id, 2.5, 0)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    planner = HybridWaypointMotionController(6, game, env)

    targets = [(0,0)]+[(random.uniform(-2, 2), random.uniform(-1.5, 1.5)) for _ in range(1000)]
    ba_targets = [(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(6)]


    for i in range(5000):
        target = targets[0]
        env.draw_point(target[0], target[1], width=10, color="GREEN")

        velocity = game.get_object_velocity(GameRobot(True, mover_id))

        next_stop = planner.path_to(targets[0], mover_id)

        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame  
        r = friendly_robots[mover_id]

        env.draw_line([(r.x, r.y), next_stop], width=2, color="PINK")
        if dist((r.x, r.y), target) < 0.05 and mag(velocity) < 0.2:
            targets.pop(0)


            randomly_spawn_robots(env, True, [mover_id])
            randomly_spawn_robots(env, False, [])

            env.draw_point(target[0], target[1], width=10, color="PINK")
            pid_oren.reset(mover_id)
            pid_2d.reset(mover_id)

        cmd = go_to_point(pid_oren, pid_2d, friendly_robots[mover_id], mover_id, next_stop, face_ball((r.x, r.y), next_stop))
        sim_robot_controller.add_robot_commands(cmd, mover_id)

        if moving:
            if i % 50 == 0:
                ba_targets = [(random.uniform(-2, 2), random.uniform(-1.5, 1.5)) for _ in range(6)]
        
            cmd_dict = {}
            for i in range(6):
                if i != mover_id:
                    cmd_dict[i] = go_to_point(pid_oren, slow_pid2d, friendly_robots[i], i,ba_targets[i], None)

            sim_robot_controller.add_robot_commands(cmd_dict)
              
        sim_robot_controller.send_robot_commands()


def randomly_spawn_robots(env: SSLStandardEnv, is_team_yellow: bool, safe_robots: List[int]):
    for i in range(6):
        if i not in safe_robots:
            env.teleport_robot(is_team_yellow, i, random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25))

if __name__ == "__main__":
    try:
        test_pathfinding(False, True)
    except KeyboardInterrupt:
        print("Exiting...")
