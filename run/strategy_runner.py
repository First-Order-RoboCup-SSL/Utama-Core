from dataclasses import replace
import time
import threading
from typing import Tuple, Optional, List
import warnings

from config.settings import MAX_CAMERAS, MAX_GAME_HISTORY, TIMESTEP, MAX_ROBOTS
from config.defaults import LEFT_START_ONE, RIGHT_START_ONE
from collections import deque
from entities.game import GameHistory, Game
from entities.data.raw_vision import RawVisionData
from entities.data.command import RobotCommand
from motion_planning.src.motion_controller import MotionController

from global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
from run.receivers import RefereeMessageReceiver, VisionReceiver
from run.refiners import (
    RobotInfoRefiner,
    PositionRefiner,
    RefereeRefiner,
    VelocityRefiner,
)
from run import GameGater
from test.common.abstract_test_manager import AbstractTestManager, TestingStatus

# from strategy.examples.strategies.one_robot_placement_strategy import (
#     RobotPlacementStrategy,
# )
from strategy.common.abstract_strategy import AbstractStrategy
from team_controller.src.controllers import (
    GRSimRobotController,
    RSimRobotController,
    RealRobotController,
    RSimPVPManager,
    RSimController,
    GRSimController,
    AbstractSimController,
)

from rsoccer_simulator.src.ssl.envs import SSLStandardEnv

import logging

logging.basicConfig(
    filename="Utama.log",
    level=logging.CRITICAL,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(
    __name__
)  # If this is within the class, or define it globally in the module


class StrategyRunner:
    """
    Main class to run the robot controller and strategy.
    Args:
        strategy (AbstractStrategy): The strategy to be used.
        my_team_is_yellow (bool): Whether the team is yellow.
        my_team_is_right (bool): Whether the team is on the right side.
        mode (str): "real", "rsim", "grism"
        exp_friendly (int): Expected number of friendly robots.
        exp_enemy (int): Expected number of enemy robots.
        opp_strategy (AbstractStrategy, optional): Opponent strategy for pvp. Defaults to None for single player.
    """

    def __init__(
        self,
        strategy: AbstractStrategy,
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        mode: str,
        exp_friendly: int,
        exp_enemy: int,
        opp_strategy: Optional[AbstractStrategy] = None,
    ):
        self.my_strategy = strategy
        self.my_team_is_yellow = my_team_is_yellow
        self.my_team_is_right = my_team_is_right
        self.mode = mode
        self.exp_friendly = exp_friendly
        self.exp_enemy = exp_enemy
        self.opp_strategy = opp_strategy
        self.logger = logging.getLogger(__name__)

        self._assert_exp_robots()
        self.rsim_env, self.sim_controller = self._load_sim_and_controller()
        self.vision_buffers, self.ref_buffer = self._setup_vision_and_referee()
        self._load_robot_control_and_pids()

        self.position_refiner = PositionRefiner()
        self.velocity_refiner = VelocityRefiner()
        self.robot_info_refiner = RobotInfoRefiner()
        # self.referee_refiner = RefereeRefiner()
        
        self.my_strategy.setup_tree()
        if self.opp_strategy:
            self.opp_strategy.setup_tree()
        
        (
            self.my_game_history,
            self.my_current_game_frame,
            self.my_game,
            self.opp_game_history,
            self.opp_current_game_frame,
            self.opp_game,
        ) = self._load_game()
        self.game_start_time = time.time()
            
        self.toggle_opp_first = False  # alternate the order of opp and friendly in run

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

    def _load_sim_and_controller(
        self,
    ) -> Tuple[Optional[SSLStandardEnv], Optional[AbstractSimController]]:
        """
        Mode "rsim": Loads the RSim environment with the expected number of robots and corresponding sim controller.
        Mode "grsim": Loads corresponding sim controller and teleports robots in GRSim to ensure the expected number of robots is met.

        Returns:
            SSLBaseEnv: The RSim environment (Otherwise None).
            AbstractSimController: The simulation controller for the environment (Otherwise None).
        """

        # Temporary placeholder values when not running rsim
        if self.opp_strategy:
            self.opp_strategy.load_rsim_env(None)
        self.my_strategy.load_rsim_env(None)
        
        if self.mode == "rsim":
            n_yellow, n_blue = map_friendly_enemy_to_colors(
                self.my_team_is_yellow, self.exp_friendly, self.exp_enemy
            )
            rsim_env = SSLStandardEnv(
                n_robots_yellow=n_yellow, n_robots_blue=n_blue, render_mode=None
            )

            if self.opp_strategy:
                self.opp_strategy.load_rsim_env(rsim_env)
            self.my_strategy.load_rsim_env(rsim_env)

            return rsim_env, RSimController(env=rsim_env)

        elif self.mode == "grsim":
            # can consider baking all of these directly into sim controller
            sim_controller = GRSimController()
            n_yellow, n_blue = map_friendly_enemy_to_colors(
                self.my_team_is_yellow, self.exp_friendly, self.exp_enemy
            )

            # Ensure the expected number of robots is met by teleporting them
            y_to_remove = [i for i in range(n_yellow, MAX_ROBOTS)]
            b_to_remove = [i for i in range(n_blue, MAX_ROBOTS)]
            for y in y_to_remove:
                sim_controller.set_robot_presence(y, True, False)
            for b in b_to_remove:
                sim_controller.set_robot_presence(b, False, False)

            y_to_keep = [i for i in range(n_yellow)]
            b_to_keep = [i for i in range(n_blue)]
            yellow_start, blue_start = map_left_right_to_colors(
                self.my_team_is_yellow,
                self.my_team_is_right,
                RIGHT_START_ONE,
                LEFT_START_ONE,
            )
            for y in y_to_keep:
                sim_controller.set_robot_presence(y, True, True)
                y_start = yellow_start[y]
                sim_controller.teleport_robot(
                    True, y, y_start[0], y_start[1], y_start[2]
                )
            for b in b_to_keep:
                sim_controller.set_robot_presence(b, False, True)
                b_start = blue_start[b]
                sim_controller.teleport_robot(
                    False, b, b_start[0], b_start[1], b_start[2]
                )
            sim_controller.teleport_ball(0, 0)

            return None, sim_controller

        else:
            return None, None

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
        if self.mode != "rsim":
            self.start_threads(vision_receiver)  # , referee_receiver)

        return vision_buffers, ref_buffer

    def _assert_exp_robots(self):
        """
        Assert the expected number of robots.
        """
        assert (
            self.exp_friendly <= MAX_ROBOTS
        ), "Expected number of friendly robots is too high."
        assert (
            self.exp_enemy <= MAX_ROBOTS
        ), "Expected number of enemy robots is too high."
        assert self.exp_friendly >= 1, "Expected number of friendly robots is too low."
        assert self.exp_enemy >= 1, "Expected number of enemy robots is too low."

        assert self.my_strategy.assert_exp_robots(
            self.exp_friendly, self.exp_enemy
        ), "Expected number of robots at runtime does not match my strategy."
        if self.opp_strategy:
            assert self.opp_strategy.assert_exp_robots(
                self.exp_enemy, self.exp_friendly
            ), "Expected number of robots at runtime does not match opponent strategy."

    def _load_robot_control_and_pids(self):
        if self.mode == "rsim":
            pvp_manager = None
            if self.opp_strategy:
                pvp_manager = RSimPVPManager(self.rsim_env)

            my_robot_controller = RSimRobotController(
                is_team_yellow=self.my_team_is_yellow,
                n_friendly=self.exp_friendly,
                env=self.rsim_env,
                pvp_manager=pvp_manager,
            )

            if self.opp_strategy:
                opp_robot_controller = RSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow,
                    n_friendly=self.exp_enemy,
                    env=self.rsim_env,
                    pvp_manager=pvp_manager,
                )
                if self.my_team_is_yellow:
                    pvp_manager.load_controllers(
                        my_robot_controller, opp_robot_controller
                    )
                else:
                    pvp_manager.load_controllers(
                        opp_robot_controller, my_robot_controller
                    )

        elif self.mode == "grsim":
            my_robot_controller = GRSimRobotController(
                is_team_yellow=self.my_team_is_yellow, n_friendly=self.exp_friendly
            )
            if self.opp_strategy:
                opp_robot_controller = GRSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow, n_friendly=self.exp_enemy
                )

        elif self.mode == "real":
            my_robot_controller = RealRobotController(
                is_team_yellow=self.my_team_is_yellow, n_friendly=self.exp_friendly
            )
            if self.opp_strategy:
                opp_robot_controller = RealRobotController(
                    is_team_yellow=not self.my_team_is_yellow, n_friendly=self.exp_enemy
                )

        else:
            raise ValueError("mode is invalid. Must be 'rsim', 'grsim' or 'real'")

        self.my_strategy.load_robot_controller(my_robot_controller)
        self.my_strategy.load_motion_controller(MotionController(self.mode))
        if self.opp_strategy:
            self.opp_strategy.load_robot_controller(opp_robot_controller)
            self.opp_strategy.load_motion_controller(MotionController(self.mode))

    def _load_game(self):
        my_current_game_frame, opp_current_game_frame = GameGater.wait_until_game_valid(
            self.my_team_is_yellow,
            self.my_team_is_right,
            self.exp_friendly,
            self.exp_enemy,
            self.vision_buffers,
            self.position_refiner,
            is_pvp=self.opp_strategy is not None,
            rsim_env=self.rsim_env,
        )
        my_game_history = GameHistory(MAX_GAME_HISTORY)
        my_game = Game(my_game_history, my_current_game_frame)
        if self.opp_strategy:
            opp_game_history = GameHistory(MAX_GAME_HISTORY)
            opp_game = Game(opp_game_history, opp_current_game_frame)
        else:
            opp_game_history, opp_game = None, None
        return (
            my_game_history,
            my_current_game_frame,
            my_game,
            opp_game_history,
            opp_current_game_frame,
            opp_game,
        )

    # Reset the game state and robot info in buffer
    def _reset_game(self):
        _ = self.my_strategy.robot_controller.get_robots_responses()

        (
            self.my_game_history,
            self.my_current_game_frame,
            self.my_game,
            self.opp_game_history,
            self.opp_current_game_frame,
            self.opp_game,
        ) = self._load_game()

    def _reset_robots(self):
        for i in self.my_current_game_frame.friendly_robots.keys():
            self.my_strategy.robot_controller.add_robot_commands(
                RobotCommand(0, 0, 0, 0, 0, 0), i
            )
        self.my_strategy.robot_controller.send_robot_commands()

        if self.opp_strategy and self.opp_current_game_frame:
            for i in self.opp_current_game_frame.friendly_robots.keys():
                self.opp_strategy.robot_controller.add_robot_commands(
                    RobotCommand(0, 0, 0, 0, 0, 0), i
                )
            self.opp_strategy.robot_controller.send_robot_commands()

    def run_test(
        self,
        testManager: AbstractTestManager,
        episode_timeout: float = 10.0,
        rsim_headless: bool = False,
    ) -> bool:
        """
        Run a test with the given test manager and episode timeout.
        Args:
            testManager (AbstractTestManager): The test manager to run the test.
            episode_timeout (float): The timeout for each episode in seconds.
            rsim_headless (bool): Whether to run RSim in headless mode. Defaults to False.
        """
        passed = True
        n_episodes = testManager.get_n_episodes()
        if not rsim_headless and self.rsim_env:
            self.rsim_env.render_mode = "human"
        if self.sim_controller is None:
            warnings.warn("Running test in real, defaulting to 1 episode.")
            n_episodes = 1

        testManager.load_strategies(self.my_strategy, self.opp_strategy)
        for i in range(n_episodes):
            testManager.update_episode_n(i)

            if self.sim_controller:
                testManager.reset_field(self.sim_controller, self.my_current_game_frame)
                time.sleep(0.1)  # wait for the field to reset
                # wait for the field to reset
            self._reset_game()
            episode_start_time = time.time()
            # for simplicity, we assume rsim is running in real time. May need to change this
            while True:
                if (time.time() - episode_start_time) > episode_timeout:
                    passed = False
                    self.logger.log(
                        logging.WARNING,
                        "Episode %d timed out after %f secs",
                        i,
                        episode_timeout,
                    )
                    break
                self._run_step()

                status = testManager.eval_status(self.my_current_game_frame)

                if status == TestingStatus.FAILURE:
                    passed = False
                    self._reset_robots()
                    break
                elif status == TestingStatus.SUCCESS:
                    self._reset_robots()
                    break

        return passed

    def run(self):
        if self.rsim_env:
            self.rsim_env.render_mode = "human"
        while True:
            self._run_step()
            # terminal next line print
            # print("\r")

    def _run_step(self):
        start_time = time.time()
        if self.mode == "rsim":
            vision_frames = [self.rsim_env._frame_to_observations()[0]]
        else:
            vision_frames = [
                buffer.popleft() if buffer else None for buffer in self.vision_buffers
            ]
        # referee_frame = ref_buffer.popleft()

        # alternate between opp and friendly playing
        if self.toggle_opp_first:
            if self.opp_strategy:
                self._step_game(start_time, vision_frames, True)
            self._step_game(start_time, vision_frames, False)
        else:
            self._step_game(start_time, vision_frames, False)
            if self.opp_strategy:
                self._step_game(start_time, vision_frames, True)
        self.toggle_opp_first = not self.toggle_opp_first

        end_time = time.time()

        # processing_time = end_time - start_time

        # self.logger.log(
        #     logging.WARNING if processing_time > TIMESTEP else logging.INFO,
        #     "Game loop took %f secs",
        #     processing_time,
        # )

        # Sleep to maintain FPS
        wait_time = max(0, TIMESTEP - (end_time - start_time))
        self.logger.info("Sleeping for %f secs", wait_time)
        if self.mode != "rsim":
            time.sleep(wait_time)

    def _step_game(
        self,
        iter_start_time: float,
        vision_frames: List[RawVisionData],
        running_opp: bool,
    ):
        """
        Step the game for the robot controller and strategy.
        Args:
            iter_start_time (float): The start time of the iteration.
            vision_frames (List[RawVisionData]): The vision frames.
            running_opp (bool): Whether to run the opponent strategy.
        """
        if running_opp:
            opp_responses = self.opp_strategy.robot_controller.get_robots_responses()
            game = replace(
                self.opp_current_game_frame, ts=iter_start_time - self.game_start_time
            )
            game = self.position_refiner.refine(game, vision_frames)
            game = self.velocity_refiner.refine(
                self.opp_game_history, game
            )  # , robot_frame.imu_data)
            self.opp_current_game_frame = self.robot_info_refiner.refine(
                game, opp_responses
            )
            # game = referee_refiner.refine(game, referee_frame)
            self.opp_game.add_game(game)
            self.opp_strategy.step(self.opp_game)
        else:
            my_responses = self.my_strategy.robot_controller.get_robots_responses()
            game = replace(
                self.my_current_game_frame, ts=iter_start_time - self.game_start_time
            )
            game = self.position_refiner.refine(game, vision_frames)
            game = self.velocity_refiner.refine(self.my_game_history, game)
            self.my_current_game_frame = self.robot_info_refiner.refine(
                game, my_responses
            )
            # game = referee_refiner.refine(game, referee_frame)
            self.my_game.add_game(game)
            self.my_strategy.step(self.my_game)


# if __name__ == "__main__":
# runner = StrategyRunner(
#     strategy=RobotPlacementStrategy(id=3),
#     my_team_is_yellow=True,
#     my_team_is_right=True,
#     mode="grsim",
#     exp_friendly=6,
#     exp_enemy=6,
#     opp_strategy=RobotPlacementStrategy(id=3, invert=True),
# )
# runner.run()
