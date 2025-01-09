import sys
import os
import numpy as np
import pytest
from motion_planning.src.pid.pid import TwoDPID, get_pids
from robot_control.src.skills import get_goal_centre, go_to_ball, go_to_point, align_defenders, to_defense_parametric, face_ball, velocity_to_orientation
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import find_likely_enemy_shooter, score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from robot_control.src.tests.utils import one_robot_placement, setup_pvp



def defend(pid_oren:PID, pid_2d: TwoDPID, game: Game, controller: RSimRobotController, is_yellow: bool, defender_id: int, env):
    # Assume that is_yellow <-> not is_left here # TODO : FIX
    friendly, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
    shooters_data = find_likely_enemy_shooter(enemy, balls)
    orientation = None
    tracking_ball = False
    if not shooters_data:
        target_tracking_coord = balls[0].x, balls[0].y
        # TODO game.get_ball_velocity() can return (None, None)
        if game.get_ball_velocity() is not None and None not in game.get_ball_velocity():
            orientation = velocity_to_orientation(game.get_ball_velocity())
            tracking_ball = True
    else:
        # TODO (deploy more defenders, or find closest shooter?)
        sd = shooters_data[0]
        target_tracking_coord = sd.x, sd.y
        orientation = sd.orientation

        
    real_def_pos = friendly[defender_id].x, friendly[defender_id].y
    current_def_parametric = to_defense_parametric(real_def_pos, is_left=not is_yellow)
    target = align_defenders(current_def_parametric, target_tracking_coord, orientation, not is_yellow, env)
    cmd = go_to_point(pid_oren, pid_2d, friendly[defender_id], defender_id, target, face_ball(real_def_pos, (balls[0].x, balls[0].y)))


    controller.add_robot_commands(cmd, defender_id)

    controller.send_robot_commands()

    gp = get_goal_centre(is_left=not is_yellow)
    env.draw_line([gp, (target_tracking_coord[0], target_tracking_coord[1])], width=5, color="RED" if tracking_ball else "PINK")

def attack(pid_oren: PID, pid_2d:TwoDPID, game:Game, controller: RSimRobotController, shooter_id: int, defender_is_yellow: bool) -> bool:
    cmd = score_goal(
        game,
        controller.robot_has_ball(shooter_id),
        shooter_id=shooter_id,
        pid_oren=pid_oren,
        pid_trans=pid_2d,
        is_yellow=not defender_is_yellow,
        shoot_in_left_goal=not defender_is_yellow
    )
    controller.add_robot_commands(cmd, shooter_id)
    controller.send_robot_commands()

    if game.is_ball_in_goal(not defender_is_yellow):
        print("Goal Scored at Position: ", game.get_ball_pos())
        return True
    return False

def test_single_defender(defender_id: int, shooter_id: int, defender_is_yellow: bool, headless: bool):
    game = Game()

    N_ROBOTS_YELLOW = 5
    N_ROBOTS_BLUE = 6

    env = SSLStandardEnv(n_robots_blue=N_ROBOTS_BLUE, n_robots_yellow=N_ROBOTS_YELLOW, render_mode="ansi" if headless else "human")
    env.reset()

    env.teleport_ball(2.25, -1)

    pid_oren_y, pid_2d_y = get_pids(N_ROBOTS_YELLOW)
    pid_oren_b, pid_2d_b = get_pids(N_ROBOTS_BLUE)

    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(env,  game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW)

    if defender_is_yellow:
        sim_robot_controller_attacker, sim_robot_controller_defender = sim_robot_controller_blue, sim_robot_controller_yellow
    else:
        sim_robot_controller_attacker, sim_robot_controller_defender = sim_robot_controller_yellow, sim_robot_controller_blue

    any_scored = False
    for _ in range(900):
        scored = attack(pid_oren_y, pid_2d_y, game, sim_robot_controller_attacker, shooter_id, defender_is_yellow)
        if scored:
            any_scored = True
            break
        defend(pid_oren_b, pid_2d_b, game, sim_robot_controller_defender, defender_is_yellow, defender_id, env)
    assert not any_scored




if __name__ == "__main__":
    try:
        test_single_defender(1, 2, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
