import time
from motion_planning.src.pid.pid import get_rsim_pids
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import PassBall, Play2v5
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
import logging
import random
from typing import List

logger = logging.getLogger(__name__)

MAX_TIME = 20  # in seconds
N_ROBOTS = 6
DEFENDING_ROBOTS = 5
ATTACKING_ROBOTS = 2
TARGET_COORDS = (-2, 3)
PASS_QUALITY_THRESHOLD = 2
SHOT_QUALITY_THRESHOLD = 0.4


def test_2v5_play(friendly_robot_ids: List[int], is_yellow: bool, headless: bool):
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""

    game = Game(my_team_is_yellow=is_yellow)

    if is_yellow:
        env = SSLStandardEnv(
            n_robots_blue=DEFENDING_ROBOTS,
            n_robots_yellow=ATTACKING_ROBOTS,
            render_mode="ansi" if headless else "human",
        )
    else:
        env = SSLStandardEnv(
            n_robots_yellow=DEFENDING_ROBOTS,
            n_robots_blue=ATTACKING_ROBOTS,
            render_mode="ansi" if headless else "human",
        )

    env.reset()
    env.teleport_ball(1, 1)

    pid_oren, pid_trans = get_rsim_pids(DEFENDING_ROBOTS)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )

    scored = False

    start_time = time.time()

    play_2v5_task = Play2v5(
        pid_oren,
        pid_trans,
        game,
        friendly_robot_ids,
        is_yellow,
        PASS_QUALITY_THRESHOLD,
        SHOT_QUALITY_THRESHOLD,
    )

    try:
        while True:
            env.draw_point(
                TARGET_COORDS[0],
                TARGET_COORDS[1],
                width=2,
            )

            # Check if the time limit has been exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_TIME:
                logger.info("Test Failed: Time limit exceeded.")
                assert False  # Failure

            if not scored:
                passer_cmd, receiver_cmd = play_2v5_task.enact(
                    passer_has_ball=sim_robot_controller.robot_has_ball(passer_id)
                )

                if game.is_ball_in_goal(our_side=not is_yellow):
                    logger.info(
                        f"Test Passed: Goal scored in 2v5 play in {elapsed_time:.2f} seconds."
                    )
                    scored = True
                    break

                sim_robot_controller.add_robot_commands(passer_cmd, passer_id)
                sim_robot_controller.add_robot_commands(receiver_cmd, receiver_id)
                sim_robot_controller.send_robot_commands()

        assert scored

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    test_2v5_play([0, 1], False, False)
