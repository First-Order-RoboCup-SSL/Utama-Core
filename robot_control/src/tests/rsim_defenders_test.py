import sys
import os
import numpy as np
from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.skills import go_to_ball, go_to_point
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from robot_control.src.tests.utils import one_robot_placement

def setup_pvp(env: SSLStandardEnv, game: Game, n_robots_blue: int, n_robots_yellow: int):
    pvp_manager = PVPManager(env, n_robots_blue, n_robots_yellow, game)
    sim_robot_controller_yellow = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=game, debug=True, pvp_manager=pvp_manager
    )
    sim_robot_controller_blue = RSimRobotController(
        is_team_yellow=False, env=env, game_obj=game, debug=False, pvp_manager=pvp_manager
    )
    pvp_manager.set_yellow_controller(sim_robot_controller_yellow)
    pvp_manager.set_blue_controller(sim_robot_controller_blue)
    pvp_manager.reset_env()

    return sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager

if __name__ == "__main__":
    game = Game()

    N_ROBOTS_YELLOW = 6
    N_ROBOTS_BLUE = 3

    TARGET_ROBOT = 1

    env = SSLStandardEnv(n_robots_blue=N_ROBOTS_BLUE)
    env.reset()

    env.teleport_ball(1, 1)
    pid_oren_y = PID(TIMESTEP, 8, -8, 3, 3, 0.1, num_robots=N_ROBOTS_YELLOW)
    pid_2d_y = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=N_ROBOTS_YELLOW)

    pid_oren_b = PID(TIMESTEP, 8, -8, 3, 3, 0.1, num_robots=N_ROBOTS_BLUE)
    pid_2d_b = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=N_ROBOTS_BLUE)

    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(env,  game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW)
    one_step_yellow = one_robot_placement(sim_robot_controller_yellow, True, pid_oren_y, pid_2d_y, False, TARGET_ROBOT, game)
    one_step_blue = one_robot_placement(sim_robot_controller_blue, False, pid_oren_b, pid_2d_b, True, TARGET_ROBOT, game)

    try:
        while True:
            one_step_yellow()            
            one_step_blue()

    except KeyboardInterrupt:
        print("Exiting...")
