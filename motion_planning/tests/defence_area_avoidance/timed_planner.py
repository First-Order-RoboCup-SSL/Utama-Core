from entities.game.game_object import Colour, Robot as GameRobot
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
import logging
from math import dist

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


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
    env.teleport_robot(True, mover_id, 4, 2.5)
    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    hybrid = TimedSwitchController(N_ROBOTS_YELLOW, game, Colour.YELLOW, env)
    target = (4, -2.5)

    for _ in range(5000):
        env.draw_point(4.5, 1, width=4)
        env.draw_point(4.5, -1, width=4)
        env.draw_point(3.5, 1, width=4)
        env.draw_point(3.5, -1, width=4)


        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, _, _ = latest_frame

        r = friendly_robots[mover_id]
        velocity = game.get_object_velocity(GameRobot(True, mover_id))

        next_stop = hybrid.path_to(target, mover_id, temporary_obstacles_enum=TempObstacleType.FIELD)

        env.draw_point(target[0], target[1], width=10, color="GREEN")

        if dist((r.x, r.y), target) < 0.05 and (
            velocity is None or mag(velocity) < 0.2
        ):
            print("REACHED")
            break

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

if __name__ == "__main__":
    try:
        test_pathfinding(False)
    except KeyboardInterrupt:
        print("Exiting...")
