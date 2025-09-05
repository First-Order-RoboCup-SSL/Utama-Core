import math

import numpy as np
from robot_control.src.tests.utils import one_robot_placement

from utama_core.config.settings import MAX_ANGULAR_VEL, MAX_VEL, TIMESTEP
from utama_core.entities.game import Game
from utama_core.motion_planning.src.pid.pid import (
    PID,
    PIDAccelerationLimiterWrapper,
    TwoDPID,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.team_controller.src.controllers import RSimRobotController

N_ROBOTS = 6


def get_rsim_pids_tuned(
    n_robots: int,
    kp_oren: float,
    kd_oren: float,
    kp_trans: float,
    kd_trans: float,
    ki_trans: float = 0.0,
):
    """Create tuned PID controllers for orientation and translation.

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
        num_robots=n_robots,
        integral_min=-5,
        integral_max=5,
    )
    return pid_oren, PIDAccelerationLimiterWrapper(pid_trans, max_acceleration=2)


def calc_errors(robot, ball):
    """Calculate translation and orientation errors between a robot and a ball.

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


def run_simulation(
    kp_oren: float,
    kd_oren: float,
    kp_trans: float,
    kd_trans: float,
    ki_trans: float,
    robot_to_place: int,
    is_yellow: bool,
    headless: bool,
):
    """Run a simulation (similar to test_one_robot_placement) and return performance metrics.

    Metrics measured:
      - Travel time variability (switching iterations) [translation performance]
      - Average translation error over the run
      - Average orientation error over the run

    Returns:
        tuple: (variability, avg_trans_error, avg_oren_error)
    """
    ITERS = 1500
    TARGET_OREN = math.pi / 2
    game = Game()

    ball = game.ball

    N_ROBOTS_BLUE = N_ROBOTS
    N_ROBOTS_YELLOW = N_ROBOTS

    # Set up simulation environment
    env = SSLStandardEnv(n_robots_blue=N_ROBOTS_BLUE, render_mode="ansi" if headless else "human")
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
        ki_trans,
    )

    sim_robot_controller = RSimRobotController(is_team_yellow=is_yellow, env=env, game_obj=game)

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
        env,
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
        return float("inf"), float("inf"), float("inf")

    # Compute travel time variability based on switch events
    travel_times = [change_iters[i + 1] - change_iters[i] for i in range(len(change_iters) - 1)]
    base_travel_time = travel_times[0]
    raw_variability = np.mean([abs(tt - base_travel_time) / base_travel_time for tt in travel_times])

    switch_count = len(change_iters)
    variability = raw_variability * (1.0 / switch_count)

    # Compute average errors over all iterations
    avg_trans_error = np.mean(np.abs(trans_errors))
    avg_oren_error = np.mean(np.abs(oren_errors))

    return variability, avg_trans_error, avg_oren_error


def auto_tune_pid():
    # Define initial search boundaries (ensure all ranges are positive)
    param_ranges = {
        "kp_oren": (18.0, 23.0),
        "kd_oren": (0.0, 0.15),
        "kp_trans": (0.5, 2.5),
        "kd_trans": (0.0, 0.005),
        "ki_trans": (0.0, 0.05),
    }

    # Initialize parameters with mid-range values
    current_params = {
        "kp_oren": np.mean(param_ranges["kp_oren"]),
        "kd_oren": np.mean(param_ranges["kd_oren"]),
        "kp_trans": np.mean(param_ranges["kp_trans"]),
        "kd_trans": np.mean(param_ranges["kd_trans"]),
        "ki_trans": param_ranges["ki_trans"][0],  # Start with min Ki
    }

    # Adaptive step sizes (initial = 25% of parameter range)
    steps = {
        "kp_oren": (param_ranges["kp_oren"][1] - param_ranges["kp_oren"][0]) * 0.25,
        "kd_oren": (param_ranges["kd_oren"][1] - param_ranges["kd_oren"][0]) * 0.25,
        "kp_trans": (param_ranges["kp_trans"][1] - param_ranges["kp_trans"][0]) * 0.25,
        "kd_trans": (param_ranges["kd_trans"][1] - param_ranges["kd_trans"][0]) * 0.25,
        "ki_trans": param_ranges["ki_trans"][0] * 0.25,
    }

    best_score = float("inf")
    best_params = current_params.copy()
    stagnation_counter = 0
    min_step_size = 0.05  # Minimum step size for termination
    weights = (1.0, 1.0, 1.2)  # w_variability, w_avg_trans, w_avg_oren

    # Track parameter and score history for convergence
    param_history = {param: [] for param in param_ranges}
    score_history = []
    convergence_window = 10  # Number of iterations to check for convergence
    convergence_threshold = 0.01  # Threshold for parameter and score changes

    # Define tuning order groups
    tune_groups = [
        ["kp_oren", "kp_trans"],  # First tune all Kp terms
        ["kd_oren", "kd_trans"],  # Then tune Kd terms
        ["ki_trans"],  # Finally tune Ki terms
    ]

    while True:
        improved = False

        # Process groups in specified order
        for group in tune_groups:
            # Randomize within group to avoid directional bias
            params_to_adjust = np.random.permutation(group)

            for param in params_to_adjust:
                original_value = current_params[param]

                # Positive direction test
                current_params[param] = np.clip(original_value + steps[param], *param_ranges[param])
                score_pos = _evaluate_params(current_params, weights)

                # Negative direction test
                current_params[param] = np.clip(original_value - steps[param], *param_ranges[param])
                score_neg = _evaluate_params(current_params, weights)

                # Update logic
                if min(score_pos, score_neg) < best_score:
                    improved = True
                    if score_pos < score_neg:
                        current_params[param] = np.clip(original_value + steps[param], *param_ranges[param])
                        steps[param] = min(
                            steps[param] * 1.3,
                            0.1 * (param_ranges[param][1] - param_ranges[param][0]),
                        )
                    else:
                        current_params[param] = np.clip(original_value - steps[param], *param_ranges[param])
                        steps[param] = min(
                            steps[param] * 1.3,
                            0.1 * (param_ranges[param][1] - param_ranges[param][0]),
                        )
                    best_score = min(score_pos, score_neg)
                    best_params = current_params.copy()
                    res = {key: round(best_params[key], 4) for key in best_params}
                    print(f"New best score: {best_score:.4f}, step size: {steps[param]:.4f}, best params: {res}\n")
                    stagnation_counter = 0
                else:
                    current_params[param] = np.clip(original_value, *param_ranges[param])
                    steps[param] = max(steps[param] * 0.7, min_step_size)

        # Update parameter and score history
        for param in param_ranges:
            param_history[param].append(current_params[param])
        score_history.append(best_score)

        # Check for convergence
        if len(score_history) >= convergence_window:
            param_changes = []
            for param in param_ranges:
                old_val = param_history[param][-convergence_window]
                new_val = param_history[param][-1]

                if old_val == 0:
                    # Use absolute difference if old value is zero
                    change = abs(new_val - old_val)
                else:
                    # Use relative change otherwise
                    change = abs(new_val - old_val) / abs(old_val)
                param_changes.append(change)

            max_param_change = max(param_changes)
            score_change = abs(score_history[-1] - score_history[-convergence_window])

            if score_history[-convergence_window] != 0:
                score_change /= abs(score_history[-convergence_window])

            if score_change < convergence_threshold and max_param_change < convergence_threshold:
                print("Convergence achieved")
                break
        if not improved:
            stagnation_counter += 1
            # Random jump to escape local minima
            if stagnation_counter > 3:
                for param in param_ranges:
                    current_params[param] = np.random.uniform(*param_ranges[param])
                stagnation_counter = 0

        # Termination condition based on step sizes
        if all(step < min_step_size for step in steps.values()):
            print("Termination: Step sizes are too small.")
            break

    print("\nOptimized parameters:")
    print(f"Orientation: Kp={best_params['kp_oren']:.3f}, Kd={best_params['kd_oren']:.3f}")
    print(
        f"Translation: Kp={best_params['kp_trans']:.3f}, Kd={best_params['kd_trans']:.3f}, Ki={best_params['ki_trans']:.3f}"
    )
    print(f"Best score: {best_score:.4f}")
    return best_params


def _evaluate_params(params, weights):
    """Helper function to run simulation and calculate score."""
    variability, avg_trans_err, avg_oren_err = run_simulation(
        params["kp_oren"],
        params["kd_oren"],
        params["kp_trans"],
        params["kd_trans"],
        params["ki_trans"],
        robot_to_place=1,
        is_yellow=False,
        headless=True,
    )
    return weights[0] * variability + weights[1] * avg_trans_err + weights[2] * avg_oren_err


if __name__ == "__main__":
    try:
        auto_tune_pid()
    except KeyboardInterrupt:
        print("Exiting...")
