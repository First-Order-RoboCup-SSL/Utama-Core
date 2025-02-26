import sys
import os
import numpy as np
import math
from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import go_to_ball, go_to_point
from team_controller.src.controllers import RSimRobotController, RSimController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from robot_control.src.tests.utils import one_robot_placement
import pytest
import time

N_ROBOTS = 6

### Note: Fully Commit this file Thanks :) ###

def test_one_robot_placement(robot_to_place: int, is_yellow: bool, headless: bool):
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""

    TEST_TRAVEL_TIME_THRESH = 0.03
    TEST_RESULT_OREN_THRESH = 0.10
    TEST_EXPECTED_ITERS = 3

    ITERS = 1200
    TARGET_OREN = math.pi / 2
    game = Game(my_team_is_yellow=is_yellow)
    
    old_pos = []

    N_ROBOTS_BLUE = N_ROBOTS
    N_ROBOTS_YELLOW = N_ROBOTS

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE, render_mode="ansi" if headless else "human"
    )
    env.reset()
    
    env_controller = RSimController(env)

    # Move the other defender out of the way
    for i in range(0, 6):
        if robot_to_place != i:
            env_controller.set_robot_presence(i, is_yellow, False)
        env_controller.set_robot_presence(i, not is_yellow, False)

    env.teleport_ball(1, 0)
    
    pid_oren, pid_2d = get_rsim_pids()

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )
    
    one_step = one_robot_placement(
        sim_robot_controller,
        is_yellow,
        pid_oren,
        pid_2d,
        False,
        robot_to_place,
        game,
        TARGET_OREN,
        env,
    )

    change_iters = []
    change_orens = []

    for iter in range(ITERS):
        switch, cx, cy, co = one_step()
        old_pos.append((cx, cy))
        if old_pos and len(old_pos) > 1:
            for i in range(len(old_pos)):
                if i != 0:
                    env.draw_line([(old_pos[i][0], old_pos[i][1]), (old_pos[i-1][0], old_pos[i-1][1])], color="red")
        if switch:
            change_iters.append(iter)
            change_orens.append(co)
        if len(change_iters) > 3:
            break
    assert len(change_iters) > TEST_EXPECTED_ITERS
    travel_time_0 = change_iters[1] - change_iters[0]

    for i in range(len(change_iters) - 1):
        travel_time_i = change_iters[i + 1] - change_iters[i]
        rel_diff = abs((travel_time_i - travel_time_0)) / travel_time_0
        assert rel_diff < TEST_TRAVEL_TIME_THRESH

    # for oren in change_orens:
    #     assert abs(abs(oren) - abs(TARGET_OREN)) / TARGET_OREN < TEST_RESULT_OREN_THRESH


if __name__ == "__main__":
    try:
        test_one_robot_placement(1, False, False)
    except KeyboardInterrupt:
        print("Exiting...")
