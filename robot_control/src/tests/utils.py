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
from global_utils.math_utils import rotate_vector


def setup_pvp(
    env: SSLStandardEnv, game: Game, n_robots_blue: int, n_robots_yellow: int
):
    """Factory method to setup PVP in an RSoccer environment"""
    pvp_manager = PVPManager(env, n_robots_blue, n_robots_yellow, game)
    sim_robot_controller_yellow = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=game, pvp_manager=pvp_manager
    )
    sim_robot_controller_blue = RSimRobotController(
        is_team_yellow=False, env=env, game_obj=game, pvp_manager=pvp_manager
    )
    pvp_manager.set_yellow_controller(sim_robot_controller_yellow)
    pvp_manager.set_blue_controller(sim_robot_controller_blue)
    pvp_manager.reset_env()

    return sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager


def one_robot_placement(
    controller: RSimRobotController,
    is_yellow: bool,
    pid_oren: PID,
    pid_2d: TwoDPID,
    invert: bool,
    team_robot_id: int,
    game: Game,
    target_oren: float,
    env: SSLStandardEnv,
):
    """Implements the one robot placmement test where the robot first goes to (0, 1.5) and points upwards, then
    goes to (0, -1.5) and points downwards and repeats. This is done by returning a closure which can be called
    to advance the simulation by one step, making the robot do the next step."""

    ty = -1.5 if invert else 1.5
    tx = 0

    def one_step():
        """Closure which advances the simulation by one step"""
        nonlocal tx, ty

        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
        if latest_frame:
            friendly_robots, enemy_robots, balls = latest_frame
            bx, by = game.ball.x, game.ball.y
            cx, cy, co = friendly_robots[team_robot_id]
            error = math.dist((tx, ty), (cx, cy))

            switch = error < 0.05
            if switch:
                ty *= -1
                pid_2d.reset(team_robot_id)
                pid_oren.reset(team_robot_id)

            # changed so the robot tracks the ball while moving
            oren = np.atan2(by - cy, bx - cx)
            cmd = go_to_point(
                game,
                pid_oren,
                pid_2d,
                team_robot_id,
                (tx, ty),
                oren,
            )
            # Rotate the local forward and left velocities to the global frame
            lf_x, lf_y = rotate_vector(cmd.local_forward_vel, 0, -co)
            ll_x, ll_y = rotate_vector(0, cmd.local_left_vel, -co)
            
            # Draw the local forward vector
            env.draw_line([(cx, cy), (cx + lf_x, cy + lf_y)], color="blue")

            # Draw the local left vector
            env.draw_line([(cx, cy), (cx + ll_x, cy + ll_y)], color="blue")

            # Rotate the global velocity vector
            gx, gy = rotate_vector(cmd.local_forward_vel, cmd.local_left_vel, -co)

            # Draw the global velocity vector
            env.draw_line([(cx, cy), (gx + cx, gy + cy)], color="black", width=2)
            
            controller.add_robot_commands(cmd, team_robot_id)
            controller.send_robot_commands()
            return (switch, cx, cy, co)

    return one_step
