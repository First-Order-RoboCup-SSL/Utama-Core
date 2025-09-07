import random
import time
from math import dist
from typing import List

from robot_control.src.skills import face_ball, go_to_point, mag

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.game import Game
from utama_core.entities.game.game_object import Robot as GameRobot
from utama_core.motion_planning.src.pid.pid import get_rsim_pids
from utama_core.motion_planning.src.planning.path_planner import (
    RRTPlanner,
    point_to_tuple,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.team_controller.src.controllers import RSimRobotController


def test_pathfinding(headless: bool):
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

    sim_robot_controller = RSimRobotController(is_team_yellow=is_yellow, env=env, game_obj=game)

    planner = RRTPlanner(game)
    targets = [(0, 0)] + [(random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(1000)]
    target = targets.pop(0)

    make_wall(env, True, 0.5, -1, [mover_id], False, 2.2)
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
            if dist((r.x, r.y), waypoints[0]) < 0.4 and (velocity is None or mag(velocity) < 0.5):
                waypoints.pop(0)

        if dist((r.x, r.y), target) < 0.05 and (velocity is None or mag(velocity) < 0.2):
            print("REACHED")
            target = targets.pop(0)
            start = time.time()
            waypoints = planner.path_to(mover_id, target)
            print("TOOK: ", time.time() - start)
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


def calculate_wall_posns(x, y, safe_robots: List[int], horizontal: bool, spread_factor: int):
    return [
        (
            robot_id,
            (
                (x + spread_factor * (2 * ROBOT_RADIUS) * posn_number, y)
                if horizontal
                else (x, y + spread_factor * (2 * ROBOT_RADIUS) * posn_number)
            ),
        )
        for (posn_number, robot_id) in zip(range(6), set(range(6)) - set(safe_robots))
    ]


def make_wall(
    env: SSLStandardEnv,
    is_team_yellow: bool,
    x: int,
    y: int,
    safe_robots: List[int],
    horizontal: bool = False,
    spread_factor: int = 1.5,
):
    for robot_id, posn in calculate_wall_posns(x, y, safe_robots, horizontal, spread_factor):
        env.teleport_robot(is_team_yellow, robot_id, posn[0], posn[1])


if __name__ == "__main__":
    try:
        test_pathfinding(False)
    except KeyboardInterrupt:
        print("Exiting...")
