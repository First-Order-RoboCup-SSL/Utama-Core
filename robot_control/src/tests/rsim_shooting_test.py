from motion_planning.src.pid.pid import TwoDPID
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
import pytest

ITERS = 500
N_ROBOTS = 6

def test_shooting(shooter_id: int, is_yellow: bool, headless: bool):
    """When the tests are run with pytest, these parameters are filled in
       based on whether we are in full or quick test mode (see conftest.py)"""

    game = Game()

    # Shooting team gets full complement of robots, defending team only half
    if is_yellow:
        env = SSLStandardEnv(n_robots_blue=N_ROBOTS // 2, n_robots_yellow=N_ROBOTS, render_mode="ansi" if headless else "human")
    else:
        env = SSLStandardEnv(n_robots_yellow=N_ROBOTS // 2, n_robots_blue=N_ROBOTS, render_mode="ansi" if headless else "human")

    env.reset()
    env.teleport_ball(1, 1)
    pid_oren = PID(TIMESTEP, 8, -8, 6, 0.1, 0.045, num_robots=N_ROBOTS)
    pid_trans = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=N_ROBOTS)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game, debug=False
    )

    shoot_in_left_goal = is_yellow  # Need to get this from ref info in real life; in unit testing this is fine
    goal_scored = False

    for iter in range(ITERS):
        # TODO: We should move robot_has_ball within game obj as well
        # This will do for now.
        if not goal_scored:
            cmd = score_goal(
                game,
                sim_robot_controller.robot_has_ball(shooter_id),
                shooter_id=shooter_id,
                pid_oren=pid_oren,
                pid_trans=pid_trans,
                is_yellow=is_yellow,
                shoot_in_left_goal=shoot_in_left_goal
            )

            if game.is_ball_in_goal(shoot_in_left_goal):
                print("Goal Scored at Position: ", game.get_ball_pos())
                goal_scored = True

            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()

    assert goal_scored

if __name__ == "__main__":
    try:
       test_shooting(5, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
