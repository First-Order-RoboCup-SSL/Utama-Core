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
N_ROBOTS = 7
DEFENDING_ROBOTS = 5
ATTACKING_ROBOTS = 2
TARGET_COORDS = (-2, 3)
PASS_QUALITY_THRESHOLD = 1.2
SHOT_QUALITY_THRESHOLD = 0.7
POSSESSION_CHANGE_DELAY = 2  # Delay time in seconds


def test_2v5_play(friendly_robot_ids: List[int], is_yellow: bool, headless: bool):
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

    player1_id = friendly_robot_ids[0]  # Start with robot 0
    player2_id = friendly_robot_ids[1]  # Start with robot 1
    prev_ball_possessor_id = None  # Track the previous possessor

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

            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_TIME:
                logger.info("Test Failed: Time limit exceeded.")
                assert False  # Failure

            if not scored:
                player1_has_ball = sim_robot_controller.robot_has_ball(player1_id)
                player2_has_ball = sim_robot_controller.robot_has_ball(player2_id)
                ball_possesor_id = None

                if player1_has_ball:
                    ball_possesor_id = player1_id
                elif player2_has_ball:
                    ball_possesor_id = player2_id

                last_possession_change_time = 0.0

                # Check if possession has changed
                if ball_possesor_id != prev_ball_possessor_id:
                    logger.info(
                        f"Ball possessor changed to {ball_possesor_id}. Waiting before acting..."
                    )
                    last_possession_change_time = time.time()
                    prev_ball_possessor_id = (
                        ball_possesor_id  # Update previous possessor
                    )

                # If within the delay period, send empty commands
                if time.time() - last_possession_change_time < POSSESSION_CHANGE_DELAY:
                    sim_robot_controller.send_robot_commands()  # Send no actions
                    continue  # Skip this iteration

                commands, sampled_positions, best_sample = play_2v5_task.enact(
                    ball_possesor_id
                )

                if game.is_ball_in_goal(our_side=not is_yellow):
                    logger.info(
                        f"Test Passed: Goal scored in 2v5 play in {elapsed_time:.2f} seconds."
                    )
                    scored = True
                    break

                (
                    sim_robot_controller.add_robot_commands(
                        commands.get(player1_id), player1_id
                    )
                    if commands.get(player1_id)
                    else None
                )
                (
                    sim_robot_controller.add_robot_commands(
                        commands.get(player2_id), player2_id
                    )
                    if commands.get(player2_id)
                    else None
                )

                # for rid in friendly_robot_ids:
                #    if rid in commands and rid != passer_id:
                #        receiver_id = rid  # Update receiver to the one in commands
                #        sim_robot_controller.add_robot_commands(commands[rid], rid)

                # Swap passer and receiver after a pass
                # passer_id, receiver_id = receiver_id, passer_id

                sim_robot_controller.send_robot_commands()
                if sampled_positions != None:
                    for sample in sampled_positions:
                        if sample == best_sample:
                            env.draw_point(sample.x, sample.y, "BLUE", width=2)
                        else:
                            env.draw_point(sample.x, sample.y, width=2)

        assert scored

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    test_2v5_play([0, 1], False, False)
