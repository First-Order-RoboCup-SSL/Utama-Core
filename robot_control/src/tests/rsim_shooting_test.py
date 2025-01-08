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

@pytest.mark.parametrize("shooter_id", [i for i in range(N_ROBOTS)])
@pytest.mark.parametrize("is_yellow", [False, True])
def test_shooting(shooter_id: int, is_yellow: bool):
    game = Game()

    # Shooting team gets full complement of robots, defending team only half
    if is_yellow:
        env = SSLStandardEnv(n_robots_blue=N_ROBOTS // 2, n_robots_yellow=N_ROBOTS)
    else:
        env = SSLStandardEnv(n_robots_yellow=N_ROBOTS // 2, n_robots_blue=N_ROBOTS)

    env.reset()
    env.teleport_ball(1, 1)
    pid_oren = PID(TIMESTEP, 8, -8, 6, 0.1, 0.045, num_robots=6)
    pid_trans = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=6)

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
       test_shooting(0, True)
    except KeyboardInterrupt:
        print("Exiting...")
