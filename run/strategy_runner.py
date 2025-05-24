from dataclasses import replace
import time
import threading
from typing import Tuple, Optional, List
import warnings

from config.settings import MAX_CAMERAS, MAX_GAME_HISTORY, TIMESTEP, MAX_ROBOTS
from collections import deque
from entities.game import PastGame, PresentFutureGame, Game
from entities.data.raw_vision import RawVisionData
from motion_planning.src.pid.pid import (
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)
from receivers.referee_receiver import RefereeMessageReceiver
from refiners.robot_info import RobotInfoRefiner
from refiners.position import PositionRefiner


# from refiners.referee import RefereeRefiner
from refiners.velocity import VelocityRefiner
from receivers.vision_receiver import VisionReceiver
from run import GameGater, AbstractTestManager, TestStatus

# from strategy.startup_strategy import StartupStrategy
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.behaviour_trees.behaviours.dummy_behaviour import DummyBehaviour
from strategy.examples.startup_strategy import StartupStrategy
from strategy.examples.one_robot_placement_strategy import RobotPlacementStrategy
from strategy.abstract_strategy import AbstractStrategy
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
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # If this is within the class, or define it globally in the module

class StrategyRunner:
    """
    Main class to run the robot controller and strategy.
    Args:
        strategy (AbstractStrategy): The strategy to be used.
        my_team_is_yellow (bool): Whether the team is yellow.
        my_team_is_right (bool): Whether the team is on the right side.
        mode (str): "real", "rsim", "grism"
        exp_friendly (bool): Expected number of friendly robots.
        exp_enemy (bool): Expected number of enemy robots.
        exp_ball (bool): Is ball expected?
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
        exp_ball: bool,
        opp_strategy: Optional[AbstractStrategy] = None,
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
        self.vision_buffers, self.ref_buffer = self._setup_vision_and_referee()
        self.rsim_env, self.sim_controller = self._load_sim_and_controller()
        self._load_robot_control_and_pids()

        self.position_refiner = PositionRefiner()
        self.velocity_refiner = VelocityRefiner()
        self.robot_info_refiner = RobotInfoRefiner()
        # self.referee_refiner = RefereeRefiner()
        (
            self.my_past_game,
            self.my_game,
            self.my_present_future_game,
            self.opp_past_game,
            self.opp_game,
            self.opp_present_future_game,
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

            return rsim_env, RSimController(env=rsim_env)

        elif self.mode == "grsim":
            sim_controller = GRSimController()
            if self.my_team_is_yellow:
                n_yellow = self.exp_friendly
                n_blue = self.exp_enemy
            else:
                n_yellow = self.exp_enemy
                n_blue = self.exp_friendly
            y_to_remove = [i for i in range(n_yellow, MAX_ROBOTS)]
            b_to_remove = [i for i in range(n_blue, MAX_ROBOTS)]
            for y in y_to_remove:
                sim_controller.set_robot_presence(y, True, False)
            for b in b_to_remove:
                sim_controller.set_robot_presence(b, False, False)
            time.wait(1000)

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
        opp_robot_controller, opp_pid_trans, opp_pid_oren = None, None, None
        if self.mode == "rsim":
            pvp_manager = None
            if self.opp_strategy is not None:
                pvp_manager = RSimPVPManager(self.rsim_env)
            my_robot_controller = RSimRobotController(
                is_team_yellow=self.my_team_is_yellow,
                env=self.rsim_env,
                pvp_manager=pvp_manager,
            )
            my_pid_oren, my_pid_trans = get_rsim_pids()
            if self.opp_strategy:
                opp_robot_controller = RSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow,
                    env=self.rsim_env,
                    pvp_manager=pvp_manager,
                )
                opp_pid_oren, opp_pid_trans = get_rsim_pids()

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

        elif self.mode == "grsim":
            my_robot_controller = GRSimRobotController(
                is_team_yellow=self.my_team_is_yellow
            )
            my_pid_oren, my_pid_trans = get_grsim_pids()
            if self.opp_strategy:
                opp_robot_controller = GRSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow
                )
                opp_pid_oren, opp_pid_trans = get_grsim_pids()

        elif self.mode == "real":
            my_robot_controller = RealRobotController(
                is_team_yellow=self.my_team_is_yellow
            )
            my_pid_oren, my_pid_trans = get_real_pids()
            # TODO: opponents currently not supported
            if self.opp_strategy:
                opp_robot_controller = RealRobotController(
                    is_team_yellow=not self.my_team_is_yellow
                )
                opp_pid_oren, opp_pid_trans = get_real_pids()

        else:
            raise ValueError("mode is invalid. Must be 'rsim', 'grsim' or 'real'")

        self.my_strategy.load_robot_controller(my_robot_controller)
        self.my_strategy.load_pids(my_pid_oren, my_pid_trans)
        if self.opp_strategy:
            self.opp_strategy.load_robot_controller(opp_robot_controller)
            self.opp_strategy.load_pids(opp_pid_oren, opp_pid_trans)

    def _load_game(self):
        my_game, opp_game = GameGater.wait_until_game_valid(
            self.my_team_is_yellow,
            self.my_team_is_right,
            self.exp_friendly,
            self.exp_enemy,
            self.exp_ball,
            self.vision_buffers,
            self.position_refiner,
            is_pvp=self.opp_strategy is not None,
            rsim_env=self.rsim_env,
        )
        my_past_game = PastGame(MAX_GAME_HISTORY)
        my_present_future_game = PresentFutureGame(my_past_game, my_game)
        if self.opp_strategy:
            opp_past_game = PastGame(MAX_GAME_HISTORY)
            opp_present_future_game = PresentFutureGame(opp_past_game, opp_game)
        else:
            opp_past_game, opp_present_future_game = None, None
        return (
            my_past_game,
            my_game,
            my_present_future_game,
            opp_past_game,
            opp_game,
            opp_present_future_game,
        )

    def run_test(
        self, testManager: AbstractTestManager, episode_timeout: float
    ) -> bool:
        if self.sim_controller is None:
            raise TypeError(
                f"cannot run test on {self.mode} mode. No sim_controller loaded."
            )
        passed = True
        n_episodes = testManager.get_n_episodes()
        testManager.load_strategies(self.my_strategy, self.opp_strategy)
        # testManager.load_game(self.my_game)
        for i in range(n_episodes):
            testManager.update_episode_n(i)
            testManager.reset_field(self.sim_controller, self.my_game)
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

                status = testManager.eval_status(self.my_game)

                if status == TestStatus.FAILURE:
                    passed = False
                    break
                elif status == TestStatus.SUCCESS:
                    break

        testManager.reset_field(self.sim_controller, self.my_game)
        return passed

    def run(self):
        while True:
            self._run_step()

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
            game = replace(self.opp_game, ts=iter_start_time - self.game_start_time)
            game = self.position_refiner.refine(game, vision_frames)
            game = self.velocity_refiner.refine(
                self.opp_past_game, game
            )  # , robot_frame.imu_data)
            self.opp_game = self.robot_info_refiner.refine(game, opp_responses)
            # game = referee_refiner.refine(game, referee_frame)
            self.opp_present_future_game.add_game(game)
            self.opp_strategy.step(self.opp_present_future_game)
        else:
            my_responses = self.my_strategy.robot_controller.get_robots_responses()
            game = replace(self.my_game, ts=iter_start_time - self.game_start_time)
            game = self.position_refiner.refine(game, vision_frames)
            game = self.velocity_refiner.refine(self.my_past_game, game)
            self.my_game = self.robot_info_refiner.refine(game, my_responses)
            # game = referee_refiner.refine(game, referee_frame)
            self.my_present_future_game.add_game(game)
            self.my_strategy.step(self.my_present_future_game)

if __name__ == "__main__":
    # bt = DummyBehaviour()
    # main(BehaviourTreeStrategy(sim_robot_controller, bt), sim_robot_controller)
    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(id=3),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=6,
        exp_enemy=6,
        exp_ball=True,
        opp_strategy=RobotPlacementStrategy(id=3, invert=True),
    )
    runner.run()
