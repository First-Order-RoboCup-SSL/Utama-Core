import cProfile
import logging
import signal
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rich.live import Live
from rich.text import Text

from utama_core.config.enums import Mode, mode_str_to_enum
from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.config.physical_constants import MAX_ROBOTS
from utama_core.config.settings import (
    FPS_PRINT_INTERVAL,
    MAX_CAMERAS,
    MAX_GAME_HISTORY,
    TIMESTEP,
)
from utama_core.data_processing.receivers import VisionReceiver
from utama_core.data_processing.refiners import (
    PositionRefiner,
    RobotInfoRefiner,
    VelocityRefiner,
)
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.raw_vision import RawVisionData
from utama_core.entities.game import Game, GameFrame, GameHistory
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
from utama_core.global_utils.math_utils import assert_valid_bounding_box
from utama_core.motion_planning.src.common.control_schemes import get_control_scheme
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.replay.replay_writer import ReplayWriter, ReplayWriterConfig
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.run import GameGater
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import (
    AbstractSimController,
    GRSimController,
    GRSimRobotController,
    RealRobotController,
    RSimController,
    RSimPVPManager,
    RSimRobotController,
)
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

logging.basicConfig(
    filename="Utama.log",
    level=logging.CRITICAL,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # If this is within the class, or define it globally in the module
logging.captureWarnings(True)


@dataclass(slots=True)
class SideRuntime:
    """Encapsulates all per-side (my team / opponent) runtime state.

    Args:
        strategy (AbstractStrategy): The strategy for this side.
        position_refiner (PositionRefiner): Position refiner for this side.
        velocity_refiner (VelocityRefiner): Velocity refiner for this side.
        robot_info_refiner (RobotInfoRefiner): Robot info refiner for this side.
        motion_controller (type[MotionController]): Motion controller factory for this side.
    """

    strategy: AbstractStrategy
    position_refiner: PositionRefiner
    velocity_refiner: VelocityRefiner
    robot_info_refiner: RobotInfoRefiner
    motion_controller: type[MotionController]

    game: Optional[Game] = field(init=False, default=None)
    game_history: Optional[GameHistory] = field(init=False, default=None)
    current_game_frame: Optional[GameFrame] = field(init=False, default=None)


class StrategyRunner:
    """Main class to run the robot controller and strategy.

    Args:
        strategy (AbstractStrategy): The strategy to be used.
        my_team_is_yellow (bool): Whether the team is yellow.
        my_team_is_right (bool): Whether the team is on the right side.
        mode (str): "real", "rsim", "grism"
        exp_friendly (int): Expected number of friendly robots.
        exp_enemy (int): Expected number of enemy robots.
        field_bounds (FieldBounds): Configuration of the field. Defaults to standard field.
        opp_strategy (AbstractStrategy, optional): Opponent strategy for pvp. Defaults to None for single player.
        control_scheme (str, optional): Name of the motion control scheme to use.
        opp_control_scheme (str, optional): Name of the opponent motion control scheme to use. If not set, uses same as friendly.
        replay_writer_config (ReplayWriterConfig, optional): Configuration for the replay writer. If unset, replay is disabled.
        print_real_fps (bool, optional): Whether to print real FPS. Defaults to False.
        profiler_name (Optional[str], optional): Enables and sets profiler name. Defaults to None which disables profiler.
        rsim_noise (RsimGaussianNoise, optional): When running in rsim, add Gaussian noise to balls and robots with the
            given standard deviation. The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
            Defaults to 0 for each.
        rsim_vanishing (float, optional): When running in rsim, cause robots and ball to vanish with the given probability.
            Defaults to 0.
        filtering (bool, optional): Turn on Kalman filtering. Defaults to true.
    """

    def __init__(
        self,
        strategy: AbstractStrategy,
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        mode: str,
        exp_friendly: int,
        exp_enemy: int,
        field_bounds: FieldBounds = Field.FULL_FIELD_BOUNDS,
        opp_strategy: Optional[AbstractStrategy] = None,
        control_scheme: str = "pid",  # This is also the default control scheme used in the motion planning tests
        opp_control_scheme: Optional[str] = None,
        replay_writer_config: Optional[ReplayWriterConfig] = None,
        print_real_fps: bool = False,  # Turn this on for RSim
        profiler_name: Optional[str] = None,
        rsim_noise: RsimGaussianNoise = RsimGaussianNoise(),
        rsim_vanishing: float = 0,
        filtering: bool = True,
    ):
        self.logger = logging.getLogger(__name__)

        self.my_team_is_yellow = my_team_is_yellow
        self.my_team_is_right = my_team_is_right
        self.mode: Mode = self._load_mode(mode)
        self.exp_friendly = exp_friendly
        self.exp_enemy = exp_enemy
        self.field_bounds = field_bounds

        self.vision_buffers, self.ref_buffer = self._setup_vision_and_referee()

        assert_valid_bounding_box(self.field_bounds)

        self.my, self.opp = self._setup_sides_data(
            strategy, opp_strategy, filtering, control_scheme, opp_control_scheme
        )

        ### functions below rely on self.my and self.opp ###

        self.rsim_env, self.sim_controller = self._load_sim(rsim_noise, rsim_vanishing)
        self._assert_exp_robots(exp_friendly, exp_enemy)

        self._load_robot_controllers()

        self._load_game()

        self._assert_exp_goals()

        self.toggle_opp_first = False  # used to alternate the order of opp and friendly in run

        # Replay Writer
        self.replay_writer = (
            ReplayWriter(replay_writer_config, my_team_is_yellow, exp_friendly, exp_enemy)
            if replay_writer_config
            else None
        )

        # FPS Printing
        self.num_frames_elapsed = 0
        self.elapsed_time = 0.0
        self.print_real_fps = print_real_fps
        if print_real_fps:
            self._fps_live = Live(auto_refresh=False)
            self._fps_live.start()  # manually control it so it never overrides prints
        else:
            self._fps_live = None

        # Profiler setup
        self.profiler_name = profiler_name
        self.profiler = cProfile.Profile() if profiler_name else None
        self._stop_event = threading.Event()

    def _handle_sigint(self, sig, frame):
        self._stop_event.set()

    def _load_mode(self, mode_str: str) -> Mode:
        """Convert a mode string to a Mode enum value.

        Performs case-insensitive lookup and raises a ValueError if the
        provided string does not map to a known Mode.

        Args:
            mode_str: Mode string (e.g. "rsim", "grsim", "real").

        Returns:
            Corresponding Mode enum.

        Raises:
            ValueError: If mode_str is not a recognized mode.
        """
        mode = mode_str_to_enum.get(mode_str.lower())
        if mode is None:
            raise ValueError(f"Unknown mode: {mode_str}. Choose from 'rsim', 'grsim', or 'real'.")
        return mode

    def data_update_listener(self, receiver: VisionReceiver):
        """Listener function to pull vision data from a VisionReceiver.

        This method is intended to be run in a separate thread and will call
        the receiver to continuously pull game data.

        Args:
            receiver: VisionReceiver instance to pull data from.
        """
        # Start receiving game data; this will run in a separate thread.
        receiver.pull_game_data()

    def start_threads(self, vision_receiver: VisionReceiver):  # , referee_receiver):
        """Start background threads for receiving vision (and referee) data.

        Starts daemon threads so they do not prevent process exit.

        Args:
            vision_receiver: VisionReceiver to run in a background thread.
        """
        # Start the data receiving in separate threads
        vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
        # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)

        # Allows the thread to close when the main program exits
        vision_thread.daemon = True
        # referee_thread.daemon = True

        # Start both thread
        vision_thread.start()
        # referee_thread.start()

    def _setup_sides_data(
        self,
        my_strategy: AbstractStrategy,
        opp_strategy: Optional[AbstractStrategy],
        filtering: bool,
        control_scheme: str,
        opp_control_scheme: Optional[str],
    ) -> Tuple[SideRuntime, Optional[SideRuntime]]:
        """Setup the data structures for both sides (my team and opponent)
        Args:
            my_strategy (AbstractStrategy): The strategy for the friendly team.
            opp_strategy (Optional[AbstractStrategy]): The strategy for the opponent team. If None, opponent side will be None.
            filtering (bool): Whether to use filtering in the position refiners.
            control_scheme (str): Name of the motion control scheme to use for the friendly team.
            opp_control_scheme (Optional[str]): Name of the motion control scheme to use for the opponent team. If not set, uses same as friendly.

        Side effect: Initializes the SideRuntime for both friendly and opponent sides, including their strategies, refiners, and motion controllers.

        Returns:
            Tuple containing the SideRuntime for the friendly team and the opponent team (or None if no opponent strategy provided).
        """
        opp_side = None
        my_pos_ref, my_vel_ref, my_robot_ref = self._init_refiners(self.field_bounds, filtering=filtering)
        my_motion_controller = get_control_scheme(control_scheme)
        my_strategy.setup_behaviour_tree(is_opp_strat=False)
        my_side = SideRuntime(
            strategy=my_strategy,
            position_refiner=my_pos_ref,
            velocity_refiner=my_vel_ref,
            robot_info_refiner=my_robot_ref,
            motion_controller=my_motion_controller,
        )

        if opp_strategy is not None:
            opp_pos_ref, opp_vel_ref, opp_robot_ref = self._init_refiners(self.field_bounds, filtering=filtering)
            opp_motion_controller = (
                get_control_scheme(opp_control_scheme) if opp_control_scheme is not None else my_motion_controller
            )
            opp_strategy.setup_behaviour_tree(is_opp_strat=True)
            opp_side = SideRuntime(
                strategy=opp_strategy,
                position_refiner=opp_pos_ref,
                velocity_refiner=opp_vel_ref,
                robot_info_refiner=opp_robot_ref,
                motion_controller=opp_motion_controller,
            )

        return my_side, opp_side

    def _load_sim(
        self,
        rsim_noise: RsimGaussianNoise,
        rsim_vanishing: float,
    ) -> Tuple[Optional[SSLStandardEnv], Optional[AbstractSimController]]:
        """Mode RSIM: Loads the RSim environment with the expected number of robots and corresponding sim controller.
        Mode GRSIM: Loads corresponding sim controller and teleports robots in GRSim to ensure the expected number of
        robots is met.

        Args:
            rsim_noise (RsimGaussianNoise, optional): When running in rsim, add Gaussian noise to balls and robots with the
                given standard deviation. The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
                Defaults to 0 for each.
            rsim_vanishing (float, optional): When running in rsim, cause robots and ball to vanish with the given probability.
                Defaults to 0.

        Returns:
            SSLBaseEnv: The RSim environment (Otherwise None).
            AbstractSimController: The simulation controller for the environment (Otherwise None).
        """
        if self.mode == Mode.RSIM:
            n_yellow, n_blue = map_friendly_enemy_to_colors(self.my_team_is_yellow, self.exp_friendly, self.exp_enemy)
            rsim_env = SSLStandardEnv(
                n_robots_yellow=n_yellow,
                n_robots_blue=n_blue,
                render_mode=None,
                gaussian_noise=rsim_noise,
                vanishing=rsim_vanishing,
            )

            if self.opp:
                self.opp.strategy.load_rsim_env(rsim_env)
            self.my.strategy.load_rsim_env(rsim_env)

            return rsim_env, RSimController(env=rsim_env)

        elif self.mode == Mode.GRSIM:
            # can consider baking all of these directly into sim controller
            sim_controller = GRSimController()
            n_yellow, n_blue = map_friendly_enemy_to_colors(self.my_team_is_yellow, self.exp_friendly, self.exp_enemy)

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
                sim_controller.teleport_robot(True, y, y_start[0], y_start[1], y_start[2])
            for b in b_to_keep:
                sim_controller.set_robot_presence(b, False, True)
                b_start = blue_start[b]
                sim_controller.teleport_robot(False, b, b_start[0], b_start[1], b_start[2])
            sim_controller.teleport_ball(0, 0)

            return None, sim_controller

        else:
            return None, None

    def _setup_vision_and_referee(self) -> Tuple[deque, deque]:
        """Setup the vision and referee buffers.

        Returns:
            tuple: Vision and referee buffers.
        """
        vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
        ref_buffer = deque(maxlen=1)
        # referee_receiver = RefereeMessageReceiver(ref_buffer, debug=False)
        vision_receiver = VisionReceiver(vision_buffers)
        if self.mode != Mode.RSIM:
            self.start_threads(vision_receiver)  # , referee_receiver)

        return vision_buffers, ref_buffer

    def _assert_exp_robots(
        self,
        exp_friendly: int,
        exp_enemy: int,
    ):
        """Assert the expected number of robots."""
        assert exp_friendly <= MAX_ROBOTS, "Expected number of friendly robots is too high."
        assert exp_enemy <= MAX_ROBOTS, "Expected number of enemy robots is too high."
        assert exp_friendly >= 1, "Expected number of friendly robots is too low."
        assert exp_enemy >= 0, "Expected number of enemy robots is too low."

        assert self.my.strategy.assert_exp_robots(
            exp_friendly, exp_enemy
        ), "Expected number of robots at runtime does not match my strategy."
        if self.opp:
            assert self.opp.strategy.assert_exp_robots(
                exp_enemy, exp_friendly
            ), "Expected number of robots at runtime does not match opponent strategy."

    def _assert_exp_goals(self):
        """Assert the expected number of goals."""
        assert self.my.strategy.assert_exp_goals(
            self.my.game.field.includes_my_goal_line,
            self.my.game.field.includes_opp_goal_line,
        ), "Field does not match expected goals for my strategy."
        if self.opp:
            assert self.opp.strategy.assert_exp_goals(
                self.opp.game.field.includes_my_goal_line,
                self.opp.game.field.includes_opp_goal_line,
            ), "Field does not match expected goals for opponent strategy."

    def _load_robot_controllers(self):
        """
        Load the robot controllers and motion controllers for both friendly and opponent strategies.
        """
        if self.mode == Mode.RSIM:
            pvp_manager = None
            if self.opp:
                pvp_manager = RSimPVPManager(self.rsim_env)

            my_robot_controller = RSimRobotController(
                is_team_yellow=self.my_team_is_yellow,
                n_friendly=self.exp_friendly,
                env=self.rsim_env,
                pvp_manager=pvp_manager,
            )

            if self.opp:
                opp_robot_controller = RSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow,
                    n_friendly=self.exp_enemy,
                    env=self.rsim_env,
                    pvp_manager=pvp_manager,
                )
                if self.my_team_is_yellow:
                    pvp_manager.load_controllers(my_robot_controller, opp_robot_controller)
                else:
                    pvp_manager.load_controllers(opp_robot_controller, my_robot_controller)

        elif self.mode == Mode.GRSIM:
            my_robot_controller = GRSimRobotController(
                is_team_yellow=self.my_team_is_yellow, n_friendly=self.exp_friendly
            )
            if self.opp:
                opp_robot_controller = GRSimRobotController(
                    is_team_yellow=not self.my_team_is_yellow, n_friendly=self.exp_enemy
                )

        elif self.mode == Mode.REAL:
            my_robot_controller = RealRobotController(
                is_team_yellow=self.my_team_is_yellow, n_friendly=self.exp_friendly
            )
            if self.opp:
                opp_robot_controller = RealRobotController(
                    is_team_yellow=not self.my_team_is_yellow, n_friendly=self.exp_enemy
                )

        else:
            raise ValueError("mode is invalid. Must be 'rsim', 'grsim' or 'real'")

        self.my.strategy.load_robot_controller(my_robot_controller)
        self.my.strategy.load_motion_controller(self.my.motion_controller(self.mode, self.rsim_env))
        if self.opp:
            self.opp.strategy.load_robot_controller(opp_robot_controller)
            self.opp.strategy.load_motion_controller(self.opp.motion_controller(self.mode, self.rsim_env))

    def _init_refiners(
        self,
        field_bounds: FieldBounds,
        filtering: bool,
    ) -> tuple[PositionRefiner, VelocityRefiner, RobotInfoRefiner]:
        """
        Initialize the position, velocity, and robot info refiners.
        Args:
            field_bounds (FieldBounds): The bounds of the field.
            filtering (bool): Whether to use filtering in the position refiner.
        Returns:
            tuple: The initialized PositionRefiner, VelocityRefiner, and RobotInfoRefiner.
        """
        position_refiner = PositionRefiner(
            field_bounds,
            filtering=filtering,
        )
        velocity_refiner = VelocityRefiner()
        robot_info_refiner = RobotInfoRefiner()

        return position_refiner, velocity_refiner, robot_info_refiner

    def _load_game(self):
        """
        Load the game state for both friendly and opponent strategies after waiting for valid game data with GameGater.

        Side effect: Populates game, game_history and current_game_frame on self.my (and self.opp if present).
        """
        my_current_game_frame, opp_current_game_frame = GameGater.wait_until_game_valid(
            self.my_team_is_yellow,
            self.my_team_is_right,
            self.exp_friendly,
            self.exp_enemy,
            self.vision_buffers,
            self.my.position_refiner,
            is_pvp=self.opp is not None,
            rsim_env=self.rsim_env,
        )

        self.my.position_refiner.start_filtering()
        if self.opp:
            self.opp.position_refiner.start_filtering()

        my_field = Field(self.my_team_is_right, self.field_bounds)
        self.my.game_history = GameHistory(MAX_GAME_HISTORY)
        self.my.game = Game(self.my.game_history, my_current_game_frame, field=my_field)
        self.my.current_game_frame = my_current_game_frame

        if self.opp:
            opp_field = Field(not self.my_team_is_right, self.field_bounds)
            self.opp.game_history = GameHistory(MAX_GAME_HISTORY)
            self.opp.game = Game(self.opp.game_history, opp_current_game_frame, field=opp_field)
            self.opp.current_game_frame = opp_current_game_frame

        self.my.strategy.load_game(self.my.game)
        if self.opp:
            self.opp.strategy.load_game(self.opp.game)

    # Reset the game state and robot info in buffer
    def _reset_game(self):
        """Reload game state by waiting for valid frames and reinitializing Game objects.

        Calls into the same loading logic used at construction to refresh the
        current game and history objects (useful between episodes or after resets).
        """
        _ = self.my.strategy.robot_controller.get_robots_responses()

        self.my.position_refiner.reset()
        if self.opp:
            self.opp.position_refiner.reset()
        self._load_game()

    def _reset_robots(self):
        """Send zero-velocity commands to all robots to stop them.

        Ensures both friendly and opponent robots (if present) receive
        zeroed commands and that those commands are sent immediately.
        """
        for i in self.my.current_game_frame.friendly_robots.keys():
            self.my.strategy.robot_controller.add_robot_commands(RobotCommand(0, 0, 0, 0, 0, 0), i)
        self.my.strategy.robot_controller.send_robot_commands()

        if self.opp and self.opp.current_game_frame:
            for i in self.opp.current_game_frame.friendly_robots.keys():
                self.opp.strategy.robot_controller.add_robot_commands(RobotCommand(0, 0, 0, 0, 0, 0), i)
            self.opp.strategy.robot_controller.send_robot_commands()

    def _stop_robots(self, stop_command_mult: int):
        """
        Send a series of stop commands to all robots to ensure they come to a halt.
        Args:
            stop_command_mult (int): Number of times to send the stop command.
        """
        my_stop_commands = {
            robot_id: RobotCommand(0, 0, 0, 0, 0, 0) for robot_id in self.my.game.friendly_robots.keys()
        }
        if self.opp and self.opp.game:
            opp_stop_commands = {
                robot_id: RobotCommand(0, 0, 0, 0, 0, 0) for robot_id in self.opp.game.friendly_robots.keys()
            }

        for _ in range(stop_command_mult):
            self.my.strategy.robot_controller.add_robot_commands(my_stop_commands)
            self.my.strategy.robot_controller.send_robot_commands()
            if self.opp and self.opp.game:
                self.opp.strategy.robot_controller.add_robot_commands(opp_stop_commands)
                self.opp.strategy.robot_controller.send_robot_commands()

    def close(self, stop_command_mult: int = 20):
        """
        Close resources used by the StrategyRunner and stop robots if in real mode.
        Args:
            stop_command_mult (int): Number of times to send the stop command to robots.
        """
        self.logger.info("Cleaning up resources...")

        if self.mode == Mode.REAL:
            try:
                self._stop_robots(stop_command_mult)
            except Exception:
                self.logger.exception("Was unable to stop robots cleanly.")
        if self.profiler:
            self.profiler.disable()
            if self.profiler.getstats():
                self.profiler.dump_stats(f"{self.profiler_name}.prof")
        if self.replay_writer:
            self.replay_writer.close()
        if self.rsim_env:
            self.rsim_env.close()
        if self._fps_live:
            self._fps_live.stop()

    def run_test(
        self,
        test_manager: AbstractTestManager,
        episode_timeout: float = 10.0,
        rsim_headless: bool = False,
    ) -> bool:
        """Run a test with the given test manager and episode timeout.
        Args:
            test_manager (AbstractTestManager): The test manager to run the test.
            episode_timeout (float): The timeout for each episode in seconds.
            rsim_headless (bool): Whether to run RSim in headless mode. Defaults to False.
        """
        signal.signal(signal.SIGINT, self._handle_sigint)

        passed = True
        n_episodes = test_manager.n_episodes
        if not rsim_headless and self.rsim_env:
            self.rsim_env.render_mode = "human"
        if self.sim_controller is None:
            warnings.warn("Running test in real, defaulting to 1 episode.")
            n_episodes = 1

        test_manager.load_strategies(self.my.strategy, self.opp.strategy if self.opp else None)

        try:
            for i in range(n_episodes):
                test_manager.update_episode_n(i)

                if self.sim_controller:
                    test_manager.reset_field(self.sim_controller, self.my.game)
                    time.sleep(0.1)

                self._reset_game()
                episode_start_time = time.time()

                if self.profiler:
                    self.profiler.enable()

                while not self._stop_event.is_set():

                    if (time.time() - episode_start_time) > episode_timeout:
                        passed = False
                        self.logger.warning(
                            "Episode %d timed out after %f secs",
                            i,
                            episode_timeout,
                        )
                        break

                    try:
                        self._run_step()
                    except Exception:
                        if self._stop_event.is_set():
                            self.logger.info("Stopping run loop due to interrupt.")
                            break
                        else:
                            raise

                    status = test_manager.eval_status(self.my.game)

                    if status == TestingStatus.FAILURE:
                        passed = False
                        self._reset_robots()
                        break
                    elif status == TestingStatus.SUCCESS:
                        self._reset_robots()
                        break

                if self._stop_event.is_set():
                    break

                if self.profiler:
                    self.profiler.disable()

            return passed

        finally:
            self.close()

    def run(self):
        """Run the main loop, stepping the game until interrupted.

        If an RSim environment is present, it ensures rendering is on. The loop
        continues until a KeyboardInterrupt is received, after which resources
        (such as replay writer and rsim env) are closed.
        """
        signal.signal(signal.SIGINT, self._handle_sigint)

        if self.rsim_env:
            self.rsim_env.render_mode = "human"
        if self.profiler:
            self.profiler.enable()
        try:
            while not self._stop_event.is_set():
                self._run_step()
        except Exception:
            if self._stop_event.is_set():
                self.logger.info("Stopping run loop due to interrupt.")
            else:
                self.logger.exception("Exception occurred during run loop:")
                raise
        finally:
            self.close()

    def _run_step(self):
        """Perform one tick of the overall game loop.

        This collects vision frames, alternates which side runs first, steps
        the strategies, writes replay frames if enabled, and enforces timestep
        rate-limiting (sleeping when necessary).

        No return value; updates internal game state and controllers.
        """
        frame_start = time.perf_counter()
        if self.mode == Mode.RSIM:
            vision_frames = [self.rsim_env._frame_to_observations()[0]]
        else:
            vision_frames = [buffer.popleft() if buffer else None for buffer in self.vision_buffers]
        # referee_frame = ref_buffer.popleft()

        # alternate between opp and friendly playing
        if self.toggle_opp_first:
            if self.opp:
                self._step_game(vision_frames, True)
            self._step_game(vision_frames, False)
        else:
            self._step_game(vision_frames, False)
            if self.opp:
                self._step_game(vision_frames, True)
        self.toggle_opp_first = not self.toggle_opp_first

        # --- rate limiting ---
        if self.mode != Mode.RSIM:
            processing_time = time.perf_counter() - frame_start
            wait_time = max(0, TIMESTEP - processing_time)
            time.sleep(wait_time)

        # --- end of frame ---
        if self.print_real_fps:
            frame_end = time.perf_counter()
            frame_dt = frame_end - frame_start

            self.elapsed_time += frame_dt
            self.num_frames_elapsed += 1

            if self.elapsed_time >= FPS_PRINT_INTERVAL:
                fps = self.num_frames_elapsed / self.elapsed_time

                # Update the live FPS area (one line, no box)
                self._fps_live.update(Text(f"FPS: {fps:.2f}"))
                self._fps_live.refresh()

                self.elapsed_time = 0.0
                self.num_frames_elapsed = 0

    def _step_game(
        self,
        vision_frames: List[RawVisionData],
        running_opp: bool,
    ):
        """Step the game for the robot controller and strategy.

        Args:
            vision_frames (List[RawVisionData]): The vision frames.
            running_opp (bool): Whether to run the opponent strategy.
        """
        side = self.opp if running_opp else self.my

        # Pull responses from robot controller
        responses = side.strategy.robot_controller.get_robots_responses()

        # Update game frame with refined information
        new_game_frame = side.position_refiner.refine(side.current_game_frame, vision_frames)
        new_game_frame = side.velocity_refiner.refine(side.game_history, new_game_frame)  # , robot_frame.imu_data)
        new_game_frame = side.robot_info_refiner.refine(new_game_frame, responses)
        # new_game_frame = self.referee_refiner.refine(new_game_frame, responses)

        # Store updated game frame
        side.current_game_frame = new_game_frame

        # write to replay
        if self.replay_writer and (running_opp != self.replay_writer.replay_configs.is_my_perspective):
            self.replay_writer.write_frame(new_game_frame)

        side.game.add_game_frame(new_game_frame)
        side.strategy.step()
