import time
from motion_planning.src.pid.pid import get_rsim_pids
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import PassBall
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
import logging
import random

logger = logging.getLogger(__name__)

MAX_TIME = 20  # in seconds
N_ROBOTS = 6


def test_passing(passer_id: int, receiver_id: int, is_yellow: bool, headless: bool):
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""

    game = Game(my_team_is_yellow=is_yellow)

    # Shooting team gets full complement of robots, defending team only half
    if is_yellow:
        env = SSLStandardEnv(
            n_robots_blue=N_ROBOTS // 2,
            n_robots_yellow=N_ROBOTS,
            render_mode="ansi" if headless else "human",
        )
    else:
        env = SSLStandardEnv(
            n_robots_yellow=N_ROBOTS // 2,
            n_robots_blue=N_ROBOTS,
            render_mode="ansi" if headless else "human",
        )

    env.reset()
    env.teleport_ball(1, 1)

    pid_oren, pid_trans = get_rsim_pids(N_ROBOTS)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    passed = False

    start_time = time.time()

    pass_ball_task = PassBall(
        pid_oren,
        pid_trans,
        game,
        passer_id,
        receiver_id,
        (-2, -2),
    )

    try:
        while True:

            # Check if the time limit has been exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_TIME:
                logger.info("Test Failed: Time limit exceeded.")
                assert False  # Failure

            if not passed:
                passer_cmd, receiver_cmd = pass_ball_task.enact(
                    passer_has_ball=sim_robot_controller.robot_has_ball(passer_id)
                )

                if sim_robot_controller.robot_has_ball(receiver_id):
                    logger.info("Passed.")
                    passed = True
                    time.sleep(1)
                    break

                sim_robot_controller.add_robot_commands(passer_cmd, passer_id)
                sim_robot_controller.add_robot_commands(receiver_cmd, receiver_id)
                sim_robot_controller.send_robot_commands()

        assert passed

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    test_passing(4, 5, False, False)
