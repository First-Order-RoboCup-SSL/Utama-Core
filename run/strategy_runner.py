from dataclasses import replace
import time
import threading
import logging
from typing import Tuple, Optional, Callable
import warnings

from config.settings import MAX_CAMERAS, MAX_GAME_HISTORY, TIMESTEP
from collections import deque
from entities.game.past_game import PastGame
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import (
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
    PID,
    TwoDPID,
)
from receivers.referee_receiver import RefereeMessageReceiver
from refiners.has_ball import HasBallRefiner
from refiners.position import PositionRefiner


# from refiners.referee import RefereeRefiner
from refiners.velocity import VelocityRefiner
from receivers.vision_receiver import VisionReceiver
from run import GameGater, AbstractTestManager, TestStatus

# from strategy.startup_strategy import StartupStrategy
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.behaviour_trees.behaviours.dummy_behaviour import DummyBehaviour
from strategy.startup_strategy import StartupStrategy
from strategy.one_robot_placement_strategy import RobotPlacementStrategy
from strategy.strategy import Strategy
from team_controller.src.controllers import (
    GRSimRobotController,
    RSimRobotController,
    RealRobotController,
    AbstractRobotController,
    RSimPVPManager,
)

from rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class StrategyRunner:
    """
    Main class to run the robot controller and strategy.
    Args:
        strategy (Strategy): The strategy to be used.
        my_team_is_yellow (bool): Whether the team is yellow.
        my_team_is_right (bool): Whether the team is on the right side.
        mode (str): "real", "rsim", "grism"
        exp_friendly (bool): Expected number of friendly robots.
        exp_enemy (bool): Expected number of enemy robots.
        exp_ball (bool): Is ball expected?
        opp_strategy (Strategy, optional): Opponent strategy for pvp. Defaults to None for single player.
    """

    def __init__(
        self,
        strategy: Strategy,
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        mode: str,
        exp_friendly: int,
        exp_enemy: int,
        exp_ball: bool,
        opp_strategy: Strategy = None,
    ):
        self.my_strategy = strategy
        self.my_team_is_yellow = my_team_is_yellow
        self.my_team_is_right = my_team_is_right
        self.mode = mode
        self.exp_friendly = exp_friendly
        self.exp_enemy = exp_enemy
        self.exp_ball = exp_ball
        self.opp_strategy = opp_strategy
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(filename="Utama.log", level=logging.INFO, filemode="w")
        warnings.simplefilter("default", DeprecationWarning)

        self._assert_exp_robots()
        self.rsim_env = self._load_rsim_env()
        self.vision_buffers, self.ref_buffer = self._setup_vision_and_referee()
        (
            self.my_robot_controller,
            self.opp_robot_controller,
            self.pid_oren,
            self.pid_trans,
        ) = self._load_robot_control_and_pids()

        self.position_refiner = PositionRefiner()
        self.velocity_refiner = VelocityRefiner()
        # self.hasball_refiner = HasBallRefiner()
        # self.referee_refiner = RefereeRefiner()
        self.past_game, self.game, self.present_future_game = self._load_game()
        self.game_start_time = time.time()

    def data_update_listener(self, receiver: VisionReceiver):
        # Start receiving game data; this will run in a separate thread.
        receiver.pull_game_data()

    def start_threads(self, vision_receiver: VisionReceiver):  # , referee_receiver):
        # Start the data receiving in separate threads
        vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
        # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)

        # Allows the thread to close when the main program exits
        vision_thread.daemon = True
        # referee_thread.daemon = True

        # Start both thread
        vision_thread.start()
        # referee_thread.start()

    def _load_rsim_env(self) -> Optional[SSLStandardEnv]:
        """
        Load the RSim environment if the mode is "rsim".
        Returns:
            SSLBaseEnv: The RSim environment.
        """
        if self.mode == "rsim":
            if self.my_team_is_yellow:
                n_yellow = self.exp_friendly
                n_blue = self.exp_enemy
            else:
                n_yellow = self.exp_enemy
                n_blue = self.exp_friendly

            rsim_env = SSLStandardEnv(
                n_robots_yellow=n_yellow,
                n_robots_blue=n_blue,
            )

            if self.opp_strategy:
                self.opp_strategy.load_rsim_env(rsim_env)
            self.my_strategy.load_rsim_env(rsim_env)

            return rsim_env
        return None

    def _setup_vision_and_referee(self) -> Tuple[deque, deque]:
        """
        Setup the vision and referee buffers.
        Returns:
            tuple: Vision and referee buffers.
        """
        vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
        ref_buffer = deque(maxlen=1)
        # referee_receiver = RefereeMessageReceiver(ref_buffer, debug=False)
        vision_receiver = VisionReceiver(vision_buffers)
        self.start_threads(vision_receiver)  # , referee_receiver)

        return vision_buffers, ref_buffer

    def _assert_exp_robots(self):
        """
        Assert the expected number of robots.
        """
        assert self.my_strategy.assert_exp_robots(
            self.exp_friendly, self.exp_enemy
        ), "Expected number of robots at runtime does not match my strategy."
        if self.opp_strategy:
            assert self.opp_strategy.assert_exp_robots(
                self.exp_enemy, self.exp_friendly
            ), "Expected number of robots at runtime does not match opponent strategy."

    def _load_robot_control_and_pids(
        self,
    ) -> Tuple[AbstractRobotController, AbstractRobotController, PID, TwoDPID]:
        rsim_pvp_controller = None
        if self.mode == "rsim":
            pvp_manager = None
            if self.opp_strategy is not None:
                pvp_manager = RSimPVPManager(self.rsim_env)
            my_robot_controller = RSimRobotController(
                is_team_yellow=self.my_team_is_yellow,
                env=self.rsim_env,
                pvp_manager=pvp_manager,
            )
            opp_robot_controller = (
                RSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow,
                    env=self.rsim_env,
                    pvp_manager=pvp_manager,
                )
                if self.opp_strategy
                else None
            )
            # TODO: this can be removed eventually when we deprecate robot_has_ball in robot_controller
            if pvp_manager:
                if self.my_team_is_yellow:
                    pvp_manager.load_controllers(
                        my_robot_controller, opp_robot_controller
                    )
                else:
                    pvp_manager.load_controllers(
                        opp_robot_controller, my_robot_controller
                    )
            pid_oren, pid_trans = get_rsim_pids()

        elif self.mode == "grsim":
            my_robot_controller = GRSimRobotController(
                is_team_yellow=self.my_team_is_yellow
            )
            opp_robot_controller = (
                GRSimRobotController(is_team_yellow=not self.my_team_is_yellow)
                if self.opp_strategy
                else None
            )
            pid_oren, pid_trans = get_grsim_pids()

        elif self.mode == "real":
            my_robot_controller = RealRobotController(
                is_team_yellow=self.my_team_is_yellow
            )
            # TODO: opponents currently not supported
            opp_robot_controller = (
                RealRobotController(is_team_yellow=not self.my_team_is_yellow)
                if self.opp_strategy
                else None
            )
            pid_oren, pid_trans = get_real_pids()

        else:
            raise ValueError("mode is invalid. Must be 'rsim', 'grsim' or 'real'")

        self.my_strategy.load_robot_controller(my_robot_controller)
        self.my_strategy.load_pids(pid_oren, pid_trans)
        if self.opp_strategy:
            self.opp_strategy.load_robot_controller(opp_robot_controller)
            self.opp_strategy.load_pids(pid_oren, pid_trans)

        return (
            my_robot_controller,
            opp_robot_controller,
            pid_oren,
            pid_trans,
            rsim_pvp_controller,
        )

    def _load_game(self):
        past_game = PastGame(MAX_GAME_HISTORY)
        game = GameGater.wait_until_game_valid(
            self.my_team_is_yellow,
            self.my_team_is_right,
            self.exp_friendly,
            self.exp_enemy,
            self.exp_ball,
            self.vision_buffers,
            self.position_refiner,
        )
        present_future_game = PresentFutureGame(past_game, game)
        return past_game, game, present_future_game

    def run_test(
        self, testManager: AbstractTestManager, episode_timeout: float
    ) -> bool:
        passed = True
        n_episodes = testManager.get_n_episodes()
        testManager.reset_field()
        for i in range(n_episodes):
            start_time = time.time()
            # for simplicity, we assume rsim is running in real time. May need to change this
            while True:
                if (time.time() - start_time) < episode_timeout:
                    passed = False
                    self.logger.log(
                        logging.WARNING,
                        "Episode %d timed out after %f secs",
                        i,
                        episode_timeout,
                    )
                    break
                if self.mode == "rsim":
                    vision_frames = [self.rsim_env._frame_to_observations()[0]]
                else:
                    vision_frames = [
                        buffer.popleft() if buffer else None
                        for buffer in self.vision_buffers
                    ]
                # robot_frame = robot_buffer.popleft()
                # referee_frame = ref_buffer.popleft()

                game = replace(self.game, ts=start_time - self.game_start_time)
                game = self.position_refiner.refine(game, vision_frames)
                self.game = self.velocity_refiner.refine(
                    self.past_game, game
                )  # , robot_frame.imu_data)
                # game = hasball_refiner.refine(game, robot_frame.ir_data)
                # game = referee_refiner.refine(game, referee_frame)

                self.present_future_game.add_game(self.game)
                self.my_strategy.step(self.present_future_game)
                end_time = time.time()
                status = testManager.eval_status(game)

                if status == TestStatus.FAILURE:
                    passed = False
                    break
                elif status == TestStatus.SUCCESS:
                    break

                processing_time = end_time - start_time

                self.logger.log(
                    logging.WARNING if processing_time > TIMESTEP else logging.INFO,
                    "Game loop took %f secs",
                    processing_time,
                )

                # Sleep to maintain FPS
                wait_time = max(0, TIMESTEP - (end_time - start_time))
                self.logger.info("Sleeping for %f secs", wait_time)
                time.sleep(wait_time)
        return passed

    def run(self):
        while True:
            start_time = time.time()
            if self.mode == "rsim":
                vision_frames = [self.rsim_env._frame_to_observations()[0]]
            else:
                vision_frames = [
                    buffer.popleft() if buffer else None
                    for buffer in self.vision_buffers
                ]
            # robot_frame = robot_buffer.popleft()
            # referee_frame = ref_buffer.popleft()

            game = replace(self.game, ts=start_time - self.game_start_time)
            game = self.position_refiner.refine(game, vision_frames)
            self.game = self.velocity_refiner.refine(
                self.past_game, game
            )  # , robot_frame.imu_data)
            # game = hasball_refiner.refine(game, robot_frame.ir_data)
            # game = referee_refiner.refine(game, referee_frame)

            self.present_future_game.add_game(self.game)
            self.my_strategy.step(self.present_future_game)
            end_time = time.time()

            processing_time = end_time - start_time

            self.logger.log(
                logging.WARNING if processing_time > TIMESTEP else logging.INFO,
                "Game loop took %f secs",
                processing_time,
            )

            # Sleep to maintain FPS
            wait_time = max(0, TIMESTEP - (end_time - start_time))
            self.logger.info("Sleeping for %f secs", wait_time)
            time.sleep(wait_time)


if __name__ == "__main__":
    # bt = DummyBehaviour()
    # main(BehaviourTreeStrategy(sim_robot_controller, bt), sim_robot_controller)
    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(id=3),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=6,
        exp_ball=True,
    )
    runner.run()
