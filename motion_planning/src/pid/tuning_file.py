import sys
import os
import numpy as np
import math
import time
from motion_planning.src.pid.pid import TwoDPID, PID
from team_controller.src.config.settings import TIMESTEP, MAX_ANGULAR_VEL, MAX_VEL
from robot_control.src.tests.utils import one_robot_placement
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from team_controller.src.controllers import RSimRobotController

N_ROBOTS = 6

def get_rsim_pids_tuned(n_robots: int,
                        kp_oren: float,
                        kd_oren: float,
                        kp_trans: float,
                        kd_trans: float,
                        ki_trans: float = 0.0):
    """
    Create tuned PID controllers for orientation and translation.

    Args:
        n_robots (int): Number of robots.
        kp_oren (float): Proportional gain for orientation PID.
        kd_oren (float): Derivative gain for orientation PID.
        kp_trans (float): Proportional gain for translation PID (for both X and Y).
        kd_trans (float): Derivative gain for translation PID.
        ki_trans (float): Integral gain for translation PID.

    Returns:
        tuple: (pid_oren, pid_trans)
    """
    pid_oren = PID(
        TIMESTEP,
        MAX_ANGULAR_VEL,
        -MAX_ANGULAR_VEL,
        kp_oren,
        kd_oren,
        0,  # Ki for orientation is kept at 0
        num_robots=n_robots,
        integral_min=-10,
        integral_max=10,
    )
    pid_trans = TwoDPID(
        TIMESTEP,
        MAX_VEL,
        kp_trans,
        kd_trans,
        ki_trans,
        num_robots=n_robots
    )
    return pid_oren, pid_trans

def calc_errors(robot, ball):
    """
    Calculate translation and orientation errors between a robot and a ball.

    Args:
        robot (tuple): (x, y, orientation) of the robot.
        ball (tuple): (x, y) of the ball.

    Returns:
        tuple:
            - translation error
            - orientation error
    """
    rx, ry, ro = robot
    bx, by = ball
    
    if -1.5 <= ry <= 1.5:
        trans_error = rx
    elif ry > 1.5:
        trans_error = math.dist((rx, ry), (0, 1.5))
    elif ry < -1.5:
        trans_error = math.dist((rx, ry), (0, -1.5))
        
    oren_error = math.atan2(by - ry, bx - rx) - ro
    
    # print(f"Trans error: {trans_error}, Oren error: {oren_error}")s
    return trans_error, oren_error
        

def run_simulation(kp_oren: float,
                   kd_oren: float,
                   kp_trans: float,
                   kd_trans: float,
                   ki_trans: float,
                   robot_to_place: int,
                   is_yellow: bool,
                   headless: bool):
    """
    Run a simulation (similar to test_one_robot_placement) and return performance metrics.

    Metrics measured:
      - Travel time variability (switching iterations) [translation performance]
      - Average translation error over the run
      - Average orientation error over the run

    Returns:
        tuple: (variability, avg_trans_error, avg_oren_error)
    """
    ITERS = 1100
    TARGET_OREN = math.pi / 2
    game = Game()
    
    if is_yellow:
        robot = game.friendly_robots[robot_to_place]
    else:
        robot = game.enemy_robots[robot_to_place]
    ball = game.ball
    
    N_ROBOTS_BLUE = N_ROBOTS
    N_ROBOTS_YELLOW = N_ROBOTS
    
    # Set up simulation environment
    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        render_mode="ansi" if headless else "human"
    )
    env.reset()
    env.teleport_ball(1, 0)
    env.teleport_robot(is_yellow, robot_to_place, 0, 1.5, 0)

    # Create tuned PID controllers for orientation and translation
    pid_oren, pid_trans = get_rsim_pids_tuned(
        N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE,
        kp_oren,
        kd_oren,
        kp_trans,
        kd_trans,
        ki_trans
    )

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow,
        env=env,
        game_obj=game
    )
    
    # The one_robot_placement routine is assumed to return:
    # (switch, _, _, co, trans_error, oren_error)
    one_step = one_robot_placement(
        sim_robot_controller,
        is_yellow,
        pid_oren,
        pid_trans,
        False,
        robot_to_place,
        game,
        TARGET_OREN,
    )

    change_iters = []
    change_orens = []
    trans_errors = []
    oren_errors = []

    for iter in range(ITERS):
        switch, cx, cy, co = one_step()
        trans_error, oren_error = calc_errors((cx, cy, co), (ball.x, ball.y))
        trans_errors.append(trans_error)
        oren_errors.append(oren_error)
        if switch:
            change_iters.append(iter)
            change_orens.append(co)

    # If insufficient switch events, return high error metrics
    if len(change_iters) < 2:
        return float('inf'), float('inf'), float('inf')

    # Compute travel time variability based on switch events
    travel_times = [change_iters[i+1] - change_iters[i] for i in range(len(change_iters) - 1)]
    base_travel_time = travel_times[0]
    variability = np.mean([abs(tt - base_travel_time) / base_travel_time for tt in travel_times])
    
    # Compute average errors over all iterations
    avg_trans_error = np.mean(np.abs(trans_errors))
    avg_oren_error = np.mean(np.abs(oren_errors))

    return variability, avg_trans_error, avg_oren_error

def auto_tune_pid():
    """
    Perform a parameter sweep over a range of PID gains for both the orientation
    and translation controllers. The cost function is defined as a weighted sum of:
      - Travel time variability
      - Average translation error over the run
      - Average orientation error over the run

    Adjust the weights as needed for your application.
    """
    # Define search ranges for orientation PID gains.
    kp_oren_values = np.linspace(15.0, 20.0, 3)   # e.g., 15, 17.5, 20
    kd_oren_values = np.linspace(0.05, 0.15, 3)     # e.g., 0.05, 0.10, 0.15

    # Define search ranges for translation PID gains.
    kp_trans_values = np.linspace(8.0, 9.0, 3)      # e.g., 8.0, 8.5, 9.0
    kd_trans_values = np.linspace(0.015, 0.035, 3)    # e.g., 0.015, 0.025, 0.035
    ki_trans_values = [0.0, 0.05]                     # try with and without integral action

    best_score = float('inf')
    best_params = None

    # Weights for the different error components (adjust as necessary)
    w_variability = 0.4
    w_avg_trans = 1.6
    w_avg_oren = 1.0

    # Nested sweep over all parameters
    for kp_oren in kp_oren_values:
        for kd_oren in kd_oren_values:
            for kp_trans in kp_trans_values:
                for kd_trans in kd_trans_values:
                    for ki_trans in ki_trans_values:
                        variability, avg_trans_err, avg_oren_err = run_simulation(
                            kp_oren, kd_oren, kp_trans, kd_trans, ki_trans,
                            robot_to_place=1, is_yellow=False, headless=True
                        )
                        # Compute a combined cost
                        score = (w_variability * variability +
                                 w_avg_trans * avg_trans_err +
                                 w_avg_oren * avg_oren_err)
                        print(f"Testing: kp_oren={kp_oren:.3f}, kd_oren={kd_oren:.3f}, "
                              f"kp_trans={kp_trans:.3f}, kd_trans={kd_trans:.3f}, ki_trans={ki_trans:.3f} -> "
                              f"variability={variability:.4f}, "
                              f"avg_trans_err={avg_trans_err:.4f}, avg_oren_err={avg_oren_err:.4f}, score={score:.4f}")
                        if score < best_score:
                            best_score = score
                            best_params = (kp_oren, kd_oren, kp_trans, kd_trans, ki_trans)

    print("\nBest parameters found:")
    print(f"Orientation: Kp = {best_params[0]:.3f}, Kd = {best_params[1]:.3f}")
    print(f"Translation: Kp = {best_params[2]:.3f}, Kd = {best_params[3]:.3f}, Ki = {best_params[4]:.3f}")
    print(f"With cost score: {best_score:.4f}")

if __name__ == "__main__":
    try:
        auto_tune_pid()
    except KeyboardInterrupt:
        print("Exiting...")
