import sys
import os
from typing import List
import numpy as np
import pytest
import time
from entities.game.game_object import Colour, GameObject, Robot as GameRobot

from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from motion_planning.src.planning.controller import TempObstacleType, TimedSwitchController
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
from team_controller.src.config.settings import ROBOT_RADIUS
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

    env.reset()
    mover_id = 1

    env.teleport_ball(2.25, -1)
    env.teleport_robot(True, 3, 0, -5, -1)

    env.teleport_robot(True, mover_id, 2.5, 0)

    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)

    slow_pid2d = TwoDPID(TIMESTEP, 1, 2, 0.1, 0.0, num_robots=6)

    
    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    planner = TimedSwitchController(6, game, Colour.YELLOW, env)
    targets = [(0,0)]+[(random.uniform(-2, 2), random.uniform(-1.5, 1.5)) for _ in range(1000)]
    target = targets.pop(0)
    ba_targets = [(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(6)]

    # make_wall(env, True, 0.5, -1, [mover_id], True, 2.2)
    ma = 0
    avg = 0
    for i in range(5000):
        env.draw_point(target[0], target[1], width=10, color="GREEN")

        velocity = game.get_object_velocity(GameRobot(True, mover_id))
        start = time.time()
        next_stop = planner.path_to(target, mover_id, TempObstacleType.ALL)
        took = time.time()-start
        ma = max(ma, took)
        avg += took
        print("TOOK", round(took, 4), round(ma, 4), round(avg / (i+1), 4))


        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame  
        r = friendly_robots[mover_id]

        env.draw_line([(r.x, r.y), next_stop], width=2, color="PINK")
        if dist((r.x, r.y), target) < 0.05 and mag(velocity) < 0.2:
            target = targets.pop(0)


            randomly_spawn_robots(env, True, [mover_id])
            randomly_spawn_robots(env, False, [])

            env.draw_point(target[0], target[1], width=10, color="PINK")
            pid_oren.reset(mover_id)
            pid_2d.reset(mover_id)

        cmd = go_to_point(pid_oren, pid_2d, friendly_robots[mover_id], mover_id, next_stop, face_ball((r.x, r.y), next_stop))
        sim_robot_controller.add_robot_commands(cmd, mover_id)
        # time.sleep(2)

        if moving:
            if i % 50 == 0:
                ba_targets = [(random.uniform(-2, 2), random.uniform(-1.5, 1.5)) for _ in range(6)]
            cmd_dict = {}
            for i in range(6):
                if i != mover_id:

                    cmd_dict[i] = go_to_point(pid_oren, slow_pid2d, friendly_robots[i], i,ba_targets[i], None)
            sim_robot_controller.add_robot_commands(cmd_dict)
        

        # # moving up:
        # cmd_dict = {}
        # for (robot_id, posn) in calculate_wall_posns(0.5, 1, [mover_id], True, 2.2):  
        #     cmd_dict[robot_id] = go_to_point(pid_oren, pid_2d, friendly_robots[robot_id], robot_id, posn, None)
        
        # sim_robot_controller.add_robot_commands(cmd_dict)        
        sim_robot_controller.send_robot_commands()

def calculate_wall_posns(x, y, safe_robots: List[int], horizontal: bool, spread_factor: int):
    return [(robot_id, (x + spread_factor * (2 * ROBOT_RADIUS) * posn_number, y) if horizontal 
                else (x, y + spread_factor * (2 * ROBOT_RADIUS) * posn_number))
                for (posn_number, robot_id) in zip(range(6), set(range(6)) - set(safe_robots))]

def make_wall(env: SSLStandardEnv, is_team_yellow: bool, x: int, y: int, safe_robots: List[int], horizontal: bool = False, spread_factor: int = 1.5):
    for (robot_id, posn) in calculate_wall_posns(x, y, safe_robots, horizontal, spread_factor):
        env.teleport_robot(is_team_yellow, robot_id, posn[0], posn[1]) 

def randomly_spawn_robots(env: SSLStandardEnv, is_team_yellow: bool, safe_robots: List[int]):
    for i in range(6):
        if i not in safe_robots:
            env.teleport_robot(is_team_yellow, i, random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25))

if __name__ == "__main__":
    try:
        test_pathfinding(False, True)
    except KeyboardInterrupt:
        print("Exiting...")
