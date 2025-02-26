from typing import List
from entities.game.game_object import Colour, Robot as GameRobot
from motion_planning.src.planning.exit_strategies import ClosestPointExit
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import (
    go_to_point,
    mag,
    face_ball,
)
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from motion_planning.src.planning.controller import TempObstacleType, TimedSwitchController
from team_controller.src.config.settings import ROBOT_RADIUS
import random
from math import dist

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

    env.teleport_robot(True, mover_id, 4, 0.3)

    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    hybrid = TimedSwitchController(N_ROBOTS_YELLOW, game, ClosestPointExit(), Colour.YELLOW, env)
    targets = [(0, 0)] + [
        (random.uniform(-4.5, 4.5), random.uniform(-2.25, 2.25)) for _ in range(1000)
    ]
    target = targets.pop(0)

    make_wall(env, True, 0.5, -0.5, [mover_id], False, 1.5)
    for _ in range(5000):
        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame

        r = friendly_robots[mover_id]
        velocity = game.get_object_velocity(GameRobot(True, mover_id))
        next_stop = hybrid.path_to(target, mover_id, TempObstacleType.DEFENCE_ZONES)

        env.draw_point(target[0], target[1], width=10, color="GREEN")

        if dist((r.x, r.y), target) < 0.05 and (
            velocity is None or mag(velocity) < 0.2
        ):
            print("REACHED")
            target = targets.pop(0)

        cmd = go_to_point(
            pid_oren,
            pid_2d,
            friendly_robots[mover_id],
            mover_id,
            next_stop,
            face_ball((r.x, r.y), next_stop),
        )
        sim_robot_controller.add_robot_commands(cmd, mover_id)

        cmd = go_to_point(
            pid_oren,
            pid_2d,
            friendly_robots[mover_id],
            mover_id,
            next_stop,
            face_ball((r.x, r.y), next_stop),
        )

        sim_robot_controller.send_robot_commands()


def calculate_wall_posns(
    x, y, safe_robots: List[int], horizontal: bool, spread_factor: int
):
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
    for robot_id, posn in calculate_wall_posns(
        x, y, safe_robots, horizontal, spread_factor
    ):
        env.teleport_robot(is_team_yellow, robot_id, posn[0], posn[1])


if __name__ == "__main__":
    try:
        test_pathfinding(False, False)
    except KeyboardInterrupt:
        print("Exiting...")
