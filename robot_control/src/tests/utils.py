import math
import numpy as np
from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.skills import go_to_ball, go_to_point
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from team_controller.src.controllers import RSimRobotController
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager

def setup_pvp(env: SSLStandardEnv, game: Game, n_robots_blue: int, n_robots_yellow: int):
    """Factory method to setup PVP in an RSoccer environment"""
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

def one_robot_placement(controller: RSimRobotController, is_yellow: bool, pid_oren: PID, pid_2d: TwoDPID, invert: bool, team_robot_id: int, game: Game):
    """Implements the one robot placmement test where the robot first goes to (0, 1.5) and points upwards, then
       goes to (0, -1.5) and points downwards and repeats. This is done by returning a closure which can be called
       to advance the simulation by one step, making the robot do the next step. """
    
    ty = -1.5 if invert else 1.5
    tx = 0

    def one_step():
        """Closure which advances the simulation by one step"""
        nonlocal tx, ty

        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, enemy_robots, balls = latest_frame    
            cx, cy, co = friendly_robots[team_robot_id]
            error = math.dist((tx, ty), (cx, cy))
            
            switch = error  < 0.002
            if switch:
                ty *= -1
                pid_2d.reset(team_robot_id)
                pid_oren.reset(team_robot_id)

            oren =  math.pi / 2 if ty > 0 else - math.pi / 2
            cmd = go_to_point(pid_oren, pid_2d, friendly_robots[team_robot_id], team_robot_id, (tx, ty), oren)
            controller.add_robot_commands(cmd, team_robot_id)
            controller.send_robot_commands()
            return (switch, cx, cy, co)
    return one_step