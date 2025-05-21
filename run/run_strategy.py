from dataclasses import replace
import time
import threading
import logging

import warnings

from config.settings import MAX_CAMERAS, MAX_GAME_HISTORY, TIMESTEP
from collections import deque
from entities.game.past_game import PastGame
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import get_grsim_pids, get_real_pids, get_rsim_pids
from receivers.referee_receiver import RefereeMessageReceiver
from refiners.has_ball import HasBallRefiner
from refiners.position import PositionRefiner


# from refiners.referee import RefereeRefiner
from refiners.velocity import VelocityRefiner
from receivers.vision_receiver import VisionReceiver
from run import GameGater

# from strategy.startup_strategy import StartupStrategy
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.behaviour_trees.behaviours.dummy_behaviour import DummyBehaviour
from strategy.startup_strategy import StartupStrategy
from strategy.strategy import Strategy
from team_controller.src.controllers import (
    GRSimRobotController,
    RSimRobotController,
    RealRobotController,
)

from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv


logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def start_threads(vision_receiver):  # , referee_receiver):
    # Start the data receiving in separate threads
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)

    # Allows the thread to close when the main program exits
    vision_thread.daemon = True
    # referee_thread.daemon = True

    # Start both thread
    vision_thread.start()
    # referee_thread.start()


def run_strategy(
    strategy: Strategy,
    my_team_is_yellow: bool,
    my_team_is_right: bool,
    mode: str,
    exp_friendly: bool,
    exp_enemy: bool,
    exp_ball: bool,
    rsim_env: SSLBaseEnv = None,
):
    """
    Main function to run the robot controller and strategy.
    Args:
        strategy (Strategy): The strategy to be used.
        my_team_is_yellow (bool): Whether the team is yellow.
        my_team_is_right (bool): Whether the team is on the right side.
        mode (str): "real", "rsim", "grism"
        exp_friendly (bool): Expected number of friendly robots.
        exp_enemy (bool): Expected number of enemy robots.
        exp_ball (bool): Expected number of balls.
        rsim_env (SSLBaseEnv, optional): Environment for RSim. Defaults to None.
    """
    logging.basicConfig(filename="Utama.log", level=logging.INFO, filemode="w")
    warnings.simplefilter("default", DeprecationWarning)

    assert strategy.assert_exp_robots(
        exp_friendly, exp_enemy
    )  # ensure the expected number of robots matches

    if mode == "rsim":
        assert rsim_env is not None, "RSim environment must be provided for RSim mode."
        robot_controller = RSimRobotController(
            is_team_yellow=my_team_is_yellow, env=rsim_env
        )
        pid_oren, pid_trans = get_rsim_pids()
        strategy.load_rsim_env(rsim_env)
    elif mode == "grsim":
        robot_controller = GRSimRobotController(is_team_yellow=my_team_is_yellow)
        pid_oren, pid_trans = get_grsim_pids()
    elif mode == "real":
        robot_controller = RealRobotController(is_team_yellow=my_team_is_yellow)
        pid_oren, pid_trans = get_real_pids()
    else:
        raise ValueError("mode is invalid. Must be 'rsim', 'grsim' or 'real'")

    strategy.load_robot_controller(robot_controller=robot_controller)
    strategy.load_pids(pid_oren=pid_oren, pid_trans=pid_trans)

    robot_buffer = deque(
        maxlen=1
    )  # TODO: Add separate thread to read robot data when we have it
    vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
    ref_buffer = deque(maxlen=1)

    # referee_receiver = RefereeMessageReceiver(ref_buffer, debug=False)
    vision_receiver = VisionReceiver(vision_buffers)

    start_threads(vision_receiver)  # , referee_receiver)

    position_refiner = PositionRefiner()
    velocity_refiner = VelocityRefiner()

    past_game = PastGame(MAX_GAME_HISTORY)
    game = GameGater.wait_until_game_valid(
        my_team_is_yellow,
        my_team_is_right,
        exp_friendly,
        exp_enemy,
        exp_ball,
        vision_buffers,
        position_refiner,
    )

    # hasball_refiner = HasBallRefiner()
    # referee_refiner = RefereeRefiner()

    present_future_game = PresentFutureGame(past_game, game)

    game_start_time = time.time()
    while True:
        start_time = time.time()
        if mode == "rsim":
            vision_frames = [robot_controller.last_frame]
        else:
            vision_frames = [
                buffer.popleft() if buffer else None for buffer in vision_buffers
            ]
        # robot_frame = robot_buffer.popleft()
        # referee_frame = ref_buffer.popleft()

        game = replace(game, ts=start_time - game_start_time)
        game = position_refiner.refine(game, vision_frames)
        # game = velocity_refiner.refine(past_game, game)  # , robot_frame.imu_data)
        # game = hasball_refiner.refine(game, robot_frame.ir_data)
        # game = referee_refiner.refine(game, referee_frame)

        present_future_game.add_game(game)
        strategy.step(present_future_game)
        end_time = time.time()

        processing_time = end_time - start_time

        logger.log(
            logging.WARNING if processing_time > TIMESTEP else logging.INFO,
            "Game loop took %f secs",
            processing_time,
        )

        # Sleep to maintain FPS
        wait_time = max(0, TIMESTEP - (end_time - start_time))
        logger.info("Sleeping for %f secs", wait_time)
        time.sleep(wait_time)


if __name__ == "__main__":
    try:
        # bt = DummyBehaviour()
        # main(BehaviourTreeStrategy(sim_robot_controller, bt), sim_robot_controller)
        run_strategy(StartupStrategy(), True, True, "grsim", 6, 6, True)
    except KeyboardInterrupt:
        print("Exiting...")
