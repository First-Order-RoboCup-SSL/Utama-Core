import sys
import os
import numpy as np
import pytest
from entities.game.game_object import Colour, GameObject, Robot as GameRobot

from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
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
from motion_planning.src.pid.path_planner import DynamicWindowPlanner

import logging

logger = logging.getLogger(__name__)



def test_pathfinding( headless: bool
):
    game = Game()
    N_ROBOTS_YELLOW = 6
    N_ROBOTS_BLUE = 6


    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    import random

    env.reset()
    mover_id = 1

    env.teleport_ball(2.25, -1)

    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)
    
    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    
    planner = DynamicWindowPlanner(game)
    targets = [(0,0)]+[(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(1000)]
    target = targets.pop(0)
    import time
    ba_targets = [(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(6)]

    
    for i in range(5000):
        env.draw_point(target[0], target[1], width=10, color="GREEN")

        velocity = game.get_object_velocity(GameRobot(True, mover_id))

        start = time.time()
        next_stop = planner.path_to(mover_id, target)
        end = time.time()
        # print("TOOK: ", end-start)
        # print("NEXT STOP", next_stop, target)

        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame  
        r = friendly_robots[mover_id]
        from math import dist
        if dist((r.x, r.y), target) < 0.05 and mag(velocity) < 0.2:
            print("REACHED")
            target = targets.pop(0)
            for i in range(6):
                env.teleport_robot(False, i, random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25))
            for i in range(6):
                if i!=mover_id:
                    env.teleport_robot(False, i, random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25))
            env.draw_point(target[0], target[1], width=10, color="PINK")
            pid_oren.reset(mover_id)
            pid_2d.reset(mover_id)

        for x in planner.par.keys():
            if planner.par[x] is not None:
                env.draw_line([planner.point_to_tuple(x), planner.point_to_tuple(planner.par[x])], color="BLUE", width=3)
                
        for i in range(len(planner.waypoints)-1):
            a,b = planner.waypoints[i], planner.waypoints[i+1]
            env.draw_line([a,b], width=3, color="PINK")
        if len(planner.waypoints) == 1:
            env.draw_line([(r.x, r.y), planner.waypoints[0]], color="PINK", width=3)
        cmd = go_to_point(pid_oren, pid_2d, friendly_robots[mover_id], mover_id, next_stop, face_ball((r.x, r.y), next_stop))
        sim_robot_controller.add_robot_commands(cmd, mover_id)
        # time.sleep(2)

        # if i % 50 == 0:
        #     ba_targets = [(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(6)]
        # cmd_dict = {}
        # for i in range(6):
        #     if i != mover_id:

        #         cmd_dict[i] = go_to_point(pid_oren, pid_2d, friendly_robots[i], i,ba_targets[i], None)
        # sim_robot_controller.add_robot_commands(cmd_dict)
        sim_robot_controller.send_robot_commands()



if __name__ == "__main__":
    try:
        test_pathfinding(False)
    except KeyboardInterrupt:
        print("Exiting...")
