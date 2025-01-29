import sys
import os
from typing import List
import numpy as np
import pytest
from shapely import Point
from entities.game.game_object import Colour, GameObject, Robot as GameRobot
from motion_planning.src.planning.path_planner import point_to_tuple
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
from motion_planning.src.planning.path_planner import DynamicWindowPlanner, RRTPlanner
from robot_control.src.find_best_shot import ROBOT_RADIUS
import random
import logging
import time
from math import dist

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    env.teleport_robot(True, mover_id, 3.5, 0)

    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    planner = RRTPlanner(game)
    targets = [(-4.5, 3)] + [
        (random.uniform(-4.5, 4.5), random.uniform(-3, 3)) for _ in range(1000)
    ]
    target = targets.pop(0)

    randomly_spawn_robots(env, True, [])

    waypoints = None
    for i in range(5000):
        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame
        r = friendly_robots[mover_id]

        env.draw_point(target[0], target[1], width=10, color="GREEN")

        velocity = game.get_object_velocity(GameRobot(True, mover_id))

        if waypoints is None:
            waypoints = planner.path_to(mover_id, target)

        if waypoints:
            if dist((r.x, r.y), waypoints[0]) < 0.4 and (
                velocity is None or mag(velocity) < 0.5
            ):
                waypoints.pop(0)

        if dist((r.x, r.y), target) < 0.05 and (
            velocity is None or mag(velocity) < 0.2
        ):
            print("REACHED")
            target = targets.pop(0)
            start = time.time()
            waypoints = planner.path_to(mover_id, target)
            print("TOOK: ", time.time() - start)
            randomly_spawn_robots(env, True, [mover_id])

            # print(waypoints)
            # time.sleep(5)
            env.draw_point(target[0], target[1], width=10, color="PINK")
            pid_oren.reset(mover_id)
            pid_2d.reset(mover_id)

        for x in planner.par.keys():
            if planner.par[x] is not None:
                env.draw_line(
                    [point_to_tuple(x), point_to_tuple(planner.par[x])],
                    color="BLUE",
                    width=3,
                )

        if waypoints is not None:
            for i in range(len(waypoints) - 1):
                a, b = waypoints[i], waypoints[i + 1]
                env.draw_line([a, b], width=3, color="PINK")

            if len(waypoints) == 1:
                env.draw_line([(r.x, r.y), waypoints[0]], color="PINK", width=3)
        if waypoints:
            next_stop = waypoints[0]
        else:
            next_stop = target

        cmd = go_to_point(
            pid_oren,
            pid_2d,
            friendly_robots[mover_id],
            mover_id,
            next_stop,
            face_ball((r.x, r.y), next_stop),
        )
        sim_robot_controller.add_robot_commands(cmd, mover_id)

        sim_robot_controller.send_robot_commands()


def randomly_spawn_robots(
    env: SSLStandardEnv, is_team_yellow: bool, safe_robots: List[int]
):
    for i in range(6):
        if i not in safe_robots:
            env.teleport_robot(
                is_team_yellow,
                i,
                random.uniform(-4.5, 4.5),
                random.uniform(-2.25, 2.25),
            )


if __name__ == "__main__":
    try:
        test_pathfinding(False, False)
    except KeyboardInterrupt:
        print("Exiting...")
