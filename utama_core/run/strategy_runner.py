import cProfile
import logging
import signal
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, FrozenSet, List, Optional, Tuple

from rich.live import Live
from rich.text import Text

from utama_core.config.enums import Mode, mode_str_to_enum
from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.config.formations import FormationType, get_formations
from utama_core.config.physical_constants import MAX_ROBOT_ID, MAX_ROBOTS
from utama_core.config.settings import (
    FPS_PRINT_INTERVAL,
    MAX_CAMERAS,
    MAX_GAME_HISTORY,
    ROBOT_FEEDBACK_CONNECTION_TIMEOUT_SECONDS,
    TIMESTEP,
)
from utama_core.custom_referee import CustomReferee
from utama_core.data_processing.receivers import RefereeMessageReceiver, VisionReceiver
from utama_core.data_processing.refiners import (
    PositionRefiner,
    RefereeRefiner,
    RobotInfoRefiner,
    VelocityRefiner,
)
from utama_core.entities.data.command import RobotCommand, RobotResponse
from utama_core.entities.data.raw_vision import RawVisionData
from utama_core.entities.game import Game, GameFrame, GameHistory
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.global_utils.mapping_utils import (
    map_colors_to_friendly_enemy,
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
from utama_core.run.referee_source import OfficialReferee, RefereeSource
from utama_core.run.vision_stream import GameFrameRenderer, RSimVisionStreamServer
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

if TYPE_CHECKING:
    from utama_core.entities.data.referee import RefereeData

_GEOMETRY_MATCH_TOLERANCE_M = 0.001  # mm-precision integers from vision → 1 mm tolerance
_VS_KICK_THRESHOLD = 0.5  # m/s — ball speed above this triggers kick commentary

logging.basicConfig(
    filename="Utama.log",
    level=logging.CRITICAL,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # If this is within the class, or define it globally in the module
logging.captureWarnings(True)


@dataclass(slots=True)
class _RobotPortFeedbackState:
    port_id: int
    has_ball: bool
    last_seen: float


def _record_robot_feedback_responses(
    feedback_by_port_id: dict[int, _RobotPortFeedbackState],
    responses: List[RobotResponse],
    now: float,
) -> None:
    for response in responses or []:
        feedback_by_port_id[response.id] = _RobotPortFeedbackState(
            port_id=response.id,
            has_ball=bool(response.has_ball),
            last_seen=now,
        )


def _build_robot_feedback_snapshot(
    feedback_by_port_id: dict[int, _RobotPortFeedbackState],
    *,
    now: float,
    my_team_is_yellow: bool,
    yellow_cmd_to_vision_mapping: dict[int, int],
    blue_cmd_to_vision_mapping: dict[int, int],
    timeout_seconds: float = ROBOT_FEEDBACK_CONNECTION_TIMEOUT_SECONDS,
) -> list[dict]:
    snapshot = []
    port_ids = set(feedback_by_port_id) | set(yellow_cmd_to_vision_mapping) | set(blue_cmd_to_vision_mapping)
    for port_id in sorted(port_ids):
        state = feedback_by_port_id.get(port_id)
        if state is None:
            item = {
                "port_id": port_id,
                "has_ball": False,
                "connected": False,
                "age_seconds": None,
            }
        else:
            age_seconds = max(0.0, now - state.last_seen)
            item = {
                "port_id": state.port_id,
                "has_ball": state.has_ball,
                "connected": age_seconds < timeout_seconds,
                "age_seconds": age_seconds,
            }

        yellow_vision_id = yellow_cmd_to_vision_mapping.get(port_id)
        blue_vision_id = blue_cmd_to_vision_mapping.get(port_id)
        if yellow_vision_id is not None:
            item["team_color"] = "yellow"
            item["team"] = "friendly" if my_team_is_yellow else "enemy"
            item["vision_id"] = yellow_vision_id
        elif blue_vision_id is not None:
            item["team_color"] = "blue"
            item["team"] = "enemy" if my_team_is_yellow else "friendly"
            item["vision_id"] = blue_vision_id

        snapshot.append(item)
    return snapshot


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
        exp_ball (bool): Whether the ball is expected to be present.
                        Only raises error when strategy expects ball but runtime does not provide it.
                        Defaults to True.
        full_field_dims (FieldDimensions): The dimensions of the full field. Defaults to standard
            field dimensions. For gRSim/Real, this must match the actual field configured in
            gRSim/SSL-Vision — a RuntimeError is raised on the first vision packet if they differ.
        field_bounds (FieldBounds): Bounds of the subset of the full field being used. Defaults to None (ie full field).
        opp_strategy (AbstractStrategy, optional): Opponent strategy for pvp. Defaults to None for single player.
        control_scheme (str, optional): Name of the motion control scheme to use.
        opp_control_scheme (str, optional): Name of the opponent motion control scheme to use. If not set, uses same as friendly.
        replay_writer_config (ReplayWriterConfig, optional): Configuration for the replay writer. If unset, replay is disabled.
        show_live_status (bool, optional): Whether to show the live terminal status panel.
            This panel includes FPS, referee command, stage, score, time remaining,
            and optional status text. Defaults to False.
        print_real_fps (bool, optional): Deprecated alias for `show_live_status`.
        profiler_name (Optional[str], optional): Enables and sets profiler name. Defaults to None which disables profiler.
        rsim_noise (RsimGaussianNoise, optional): When running in rsim, add Gaussian noise to balls and robots with the
            given standard deviation. The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
            Defaults to 0 for each.
        rsim_vanishing (float, optional): When running in rsim, cause robots and ball to vanish with the given probability.
            Defaults to 0.
        filtering (bool, optional): Turn on Kalman filtering. Defaults to false.
        referee (RefereeSource, optional): Referee source.  Pass a ``CustomReferee``
            instance to use the in-process referee, ``OfficialReferee()`` to consume
            commands from the SSL game-controller over the network, or ``None``
            (default) to run without any referee input.
        enable_vision_stream (bool, optional): Start a browser stream that renders
            the current game frame using RSim-style graphics without opening RSim.
            Defaults to True.
        yellow_vision_to_cmd_mapping (dict[int, int], optional): Mapping from vision robot IDs to command robot IDs for the yellow team.
            Used only in real mode. In real PVP/shared-transmitter mode, mappings are required for both teams and must include all expected robots.
        blue_vision_to_cmd_mapping (dict[int, int], optional): Mapping from vision robot IDs to command robot IDs for the blue team.
            Used only in real mode. In real PVP/shared-transmitter mode, mappings are required for both teams and must include all expected robots.
        yellow_trusted_ir_robots (FrozenSet[int], optional): Vision IDs of yellow-team robots whose IR (has_ball) sensor
            is confirmed working.  Robots NOT in this set fall back to vision-proximity inference (~0.13 m).
            Pass ``None`` (default) to trust all IR sensors — remove this argument once sensors are stable.
        blue_trusted_ir_robots (FrozenSet[int], optional): Same as ``yellow_trusted_ir_robots`` for the blue team.
    """

    def __init__(
        self,
        strategy: AbstractStrategy,
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        mode: str,
        exp_friendly: int,
        exp_enemy: int,
        exp_ball: bool = True,
        full_field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
        field_bounds: Optional[FieldBounds] = None,
        opp_strategy: Optional[AbstractStrategy] = None,
        control_scheme: str = "fpp",  # This is also the default control scheme used in the motion planning tests
        opp_control_scheme: Optional[str] = None,
        replay_writer_config: Optional[ReplayWriterConfig] = None,
        show_live_status: bool = False,  # Turn this on for simulator debugging
        print_real_fps: Optional[bool] = None,
        profiler_name: Optional[str] = None,
        rsim_noise: RsimGaussianNoise = RsimGaussianNoise(),
        rsim_vanishing: float = 0,
        filtering: bool = True,
        referee: RefereeSource = None,
        formation_type: Optional[FormationType] = None,
        enable_vision_stream: bool = True,
        vision_stream_http_port: int = 8765,
        yellow_vision_to_cmd_mapping: Optional[dict[int, int]] = None,
        blue_vision_to_cmd_mapping: Optional[dict[int, int]] = None,
        yellow_trusted_ir_robots: Optional[FrozenSet[int]] = None,
        blue_trusted_ir_robots: Optional[FrozenSet[int]] = None,
    ):
        self.logger = logging.getLogger(__name__)

        self._prev_custom_ref_command: Optional[RefereeCommand] = None
        self._last_referee_data: Optional["RefereeData"] = None
        self._vs_team_names: tuple[str, str] = self._assign_team_names()
        self._vs_commentary: str = "Welcome to the match!"
        self._vs_commentary_until: float = 0.0
        self._vs_prev_score: tuple[int, int] = (0, 0)
        self._vs_prev_ball_speed: float = 0.0
        self._vs_idle_index: int = 0
        self._vs_idle_next: float = 0.0
        # (is_friendly, robot_id) -> footballer name
        self._vs_robot_names: dict[tuple[bool, int], str] = {}
        self._robot_feedback_by_port_id: dict[int, _RobotPortFeedbackState] = {}
        self._robot_feedback_snapshot: list[dict] = []
        self.my_team_is_yellow = my_team_is_yellow
        self.my_team_is_right = my_team_is_right
        self.mode: Mode = self._load_mode(mode)
        self.exp_friendly = exp_friendly
        self.exp_enemy = exp_enemy
        self.exp_ball = exp_ball
        self.formation_type = formation_type
        self.full_field_dims = full_field_dims
        self.field_bounds = field_bounds if field_bounds else full_field_dims.full_field_bounds
        self.referee: RefereeSource = self._validate_referee(self.mode, referee)

        self._stop_event = threading.Event()
        self._vision_receiver: Optional[VisionReceiver] = None
        self.vision_stream: Optional[RSimVisionStreamServer] = None
        self._vision_stream_renderer: Optional[GameFrameRenderer] = None

        if isinstance(self.referee, CustomReferee):
            from utama_core.custom_referee.geometry import RefereeGeometry

            self.referee.override_geometry(RefereeGeometry.from_field_dims(self.full_field_dims))

        self.vision_buffers, self.ref_buffer = self._setup_vision_and_referee()

        assert_valid_bounding_box(
            self.field_bounds,
            self.full_field_dims.full_field_half_length,
            self.full_field_dims.full_field_half_width,
        )
        self.referee_refiner = RefereeRefiner()

        my_trusted_ir = yellow_trusted_ir_robots if my_team_is_yellow else blue_trusted_ir_robots
        opp_trusted_ir = blue_trusted_ir_robots if my_team_is_yellow else yellow_trusted_ir_robots

        # Set self.opp to a sentinel before mapping validation so _validate_vision_to_cmd_mapping
        # can check whether an opponent strategy is present (full SideRuntime is set later).
        self.opp = opp_strategy  # temporary sentinel; overwritten by _setup_sides_data below

        # Validate and store mappings before constructing refiners so that the
        # allowlists passed to PositionRefiner are always derived from validated data.
        self.yellow_vision_to_cmd_mapping = self._validate_vision_to_cmd_mapping(
            yellow_vision_to_cmd_mapping, is_yellow=True
        )
        self.blue_vision_to_cmd_mapping = self._validate_vision_to_cmd_mapping(
            blue_vision_to_cmd_mapping, is_yellow=False
        )
        self.yellow_cmd_to_vision_mapping = {v: k for k, v in self.yellow_vision_to_cmd_mapping.items()}
        self.blue_cmd_to_vision_mapping = {v: k for k, v in self.blue_vision_to_cmd_mapping.items()}

        if self.opp and self.mode == Mode.REAL:
            self._check_no_cmd_duplicate_if_transmission_sharing(
                self.yellow_vision_to_cmd_mapping, self.blue_vision_to_cmd_mapping
            )

        # Derive per-color roster allowlists from the validated mappings (real mode only).
        # Any robot ID seen by vision that is not in the allowlist is silently dropped so that
        # stray detections from robots not in play never pollute the game state.
        # Blue filtering is only applied when there is an opponent strategy — in single-team
        # mode blue robots are tracked as enemies and must not be filtered out.
        _allowed_yellow = frozenset(self.yellow_vision_to_cmd_mapping) or None
        _allowed_blue = (frozenset(self.blue_vision_to_cmd_mapping) or None) if opp_strategy is not None else None

        self.my, self.opp = self._setup_sides_data(
            strategy,
            opp_strategy,
            filtering,
            control_scheme,
            opp_control_scheme,
            my_trusted_ir_robots=my_trusted_ir,
            opp_trusted_ir_robots=opp_trusted_ir,
            allowed_yellow_ids=_allowed_yellow,
            allowed_blue_ids=_allowed_blue,
        )

        ### functions below rely on self.my and self.opp ###

        self.rsim_env, self.sim_controller = self._load_sim(rsim_noise, rsim_vanishing)
        self._assert_exp_robots_and_ball(exp_friendly, exp_enemy, exp_ball)

        self._load_robot_controllers()

        # Remove Rsim ball. Rsim does not have the flexibilty to start without a ball.
        # this must also be done after robot controllers are loaded and env reset
        # the reset are embedded in the robot controllers so that the env can be controlled
        # with the RsimRobotController even outside the context of StrategyRunner.
        if self.rsim_env and not self.exp_ball:
            self._remove_rsim_ball()

        # Load all game related data
        self._load_game()
        self._assert_exp_goals()
        if enable_vision_stream:
            self._start_vision_stream(vision_stream_http_port)

        # Seed the custom referee's internal clocks from the first real vision
        # timestamp so all timers are on the same timebase regardless of mode
        # (rsim sim-time or grsim/real wall-time).
        # Sim modes start in FORCE_START so play begins immediately; real mode
        # starts in HALT so the operator can set up before play begins.
        if isinstance(self.referee, CustomReferee):
            initial_command = RefereeCommand.HALT if self.mode == Mode.REAL else RefereeCommand.FORCE_START
            self.referee.seed_clock(self.my.current_game_frame.ts, initial_command)
        self.my.strategy.setup_behaviour_tree(is_opp_strat=False)
        if self.opp:
            self.opp.strategy.setup_behaviour_tree(is_opp_strat=True)

        # SnapshotVisitor for real-time behaviour tree visualization
        from py_trees.visitors import SnapshotVisitor

        self._bt_snapshot = SnapshotVisitor()
        self.my.strategy.behaviour_tree.add_visitor(self._bt_snapshot)

        self.toggle_opp_first = False  # used to alternate the order of opp and friendly in run

        if print_real_fps is not None:
            warnings.warn(
                "`print_real_fps` is deprecated; use `show_live_status` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            show_live_status = print_real_fps

        # Replay Writer
        self.replay_writer = (
            ReplayWriter(replay_writer_config, my_team_is_yellow, exp_friendly, exp_enemy)
            if replay_writer_config
            else None
        )

        # Live terminal status panel
        self.num_frames_elapsed = 0
        self.elapsed_time = 0.0
        self.show_live_status = show_live_status
        self.print_real_fps = show_live_status
        if show_live_status:
            self._fps_live = Live(auto_refresh=False)
            self._fps_live.start()  # manually control it so it never overrides prints
        else:
            self._fps_live = None

        # Profiler setup
        self.profiler_name = profiler_name
        self.profiler = cProfile.Profile() if profiler_name else None

    def _start_vision_stream(self, http_port: int) -> None:
        """Start the browser stream that mirrors refined game frames."""
        try:
            self._vision_stream_renderer = GameFrameRenderer(self.full_field_dims, scale=300.0)
            self.vision_stream = RSimVisionStreamServer(
                http_port=http_port,
            )
            self.vision_stream.start()
            self._publish_vision_stream_frame()
            self.logger.info("Vision stream available at %s", self.vision_stream.url)
            print(f"Vision stream available at {self.vision_stream.url}")
        except Exception:
            self.vision_stream = None
            self._vision_stream_renderer = None
            self.logger.warning("Vision stream could not be started; continuing without browser video.", exc_info=True)

    def _validate_vision_to_cmd_mapping(self, mapping: Optional[dict[int, int]], is_yellow: bool) -> dict[int, int]:
        if self.mode == Mode.REAL:
            is_my_team_color = is_yellow == self.my_team_is_yellow
            if mapping is None:
                if self.opp:
                    raise ValueError(
                        "explicit vision_to_cmd_mapping is required for both teams in real PVP/shared-transmitter mode."
                    )
                if is_my_team_color:
                    raise ValueError(
                        "vision_to_cmd_mapping is required for the friendly team in real mode. "
                        "Provide a mapping from vision robot IDs to firmware command IDs."
                    )
                return {}

            if not isinstance(mapping, dict):
                raise TypeError(
                    f"vision_to_cmd_mapping must be a dictionary mapping vision robot IDs to command robot IDs; got {type(mapping).__name__}."
                )

            # if we are not running an opp strat, but the opponent-color mapping provided, warn it will be ignored
            if self.opp is None and self.my_team_is_yellow ^ is_yellow:
                warnings.warn(
                    "vision_to_cmd_mapping is provided but will be ignored since the opponent team is not being controlled."
                )

            # Count check: mapping must have exactly as many entries as expected robots for that team.
            # Applied in both PVP and single-team real mode; the opponent-color mapping is skipped
            # when there is no opp strategy (already warned above).
            if is_my_team_color or self.opp:
                if is_my_team_color:
                    exp_count = self.exp_friendly
                    team_label = "friendly"
                else:
                    exp_count = self.exp_enemy
                    team_label = "opponent"

                # At init time we only know how many robots to expect, not their
                # actual vision IDs (those are non-contiguous in some deployments).
                # Check count here; key-coverage against observed IDs happens in
                # _validate_mapping_covers_game_frame() after _load_game().
                if len(mapping) != exp_count:
                    raise ValueError(
                        f"vision_to_cmd_mapping for {team_label} team has {len(mapping)} entries but "
                        f"{exp_count} robots are expected. Mapping must include every robot in play."
                    )

            for vision_id, cmd_id in mapping.items():
                if not isinstance(vision_id, int) or not isinstance(cmd_id, int):
                    raise TypeError(
                        f"vision_to_cmd_mapping must map integers to integers; got key type {type(vision_id).__name__} and value type {type(cmd_id).__name__}."
                    )
                if vision_id < 0 or cmd_id < 0:
                    raise ValueError(
                        f"vision_to_cmd_mapping cannot have negative IDs; got vision ID {vision_id} and command ID {cmd_id}."
                    )
                if vision_id > MAX_ROBOT_ID:
                    raise ValueError(
                        f"vision_to_cmd_mapping cannot have vision IDs greater than {MAX_ROBOT_ID}; got vision ID {vision_id}."
                    )
                if cmd_id > 0xFF:
                    raise ValueError(
                        f"vision_to_cmd_mapping cannot have command IDs greater than 255 (1 byte limit); got command ID {cmd_id}."
                    )
            return mapping
        else:
            if mapping is not None:
                raise ValueError(
                    "vision_to_cmd_mapping should not be provided in simulation modes; robot ID mapping is only needed in real mode."
                )
            return {}

    def _check_no_cmd_duplicate_if_transmission_sharing(
        self, yellow_mapping: dict[int, int], blue_mapping: dict[int, int]
    ):
        seen = set()
        dicts = [yellow_mapping, blue_mapping]
        for d in dicts:
            for v in d.values():
                if v in seen:
                    raise ValueError(
                        f"vision_to_cmd_mapping for friendly and opponent teams cannot have overlapping command IDs since commands are transmitted together; duplicate command ID: {v}."
                    )
                seen.add(v)

    def _handle_sigint(self, sig, frame):
        if self._stop_event.is_set():
            signal.default_int_handler(sig, frame)
        self._stop_event.set()
        self._stop_fps_live()
        print("\nStopping gracefully. Press Ctrl+C again to force quit.", flush=True)

    def _stop_fps_live(self):
        if self._fps_live:
            self._fps_live.stop()
            self._fps_live = None

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

    @staticmethod
    def _validate_referee(mode: Mode, referee: RefereeSource) -> RefereeSource:
        """Validate the referee source against the current mode."""
        if referee is not None and not isinstance(referee, (OfficialReferee, CustomReferee)):
            raise TypeError(
                f"referee must be None, OfficialReferee(), or a CustomReferee instance; got {type(referee).__name__}."
            )

        if isinstance(referee, OfficialReferee) and mode == Mode.RSIM:
            raise ValueError("OfficialReferee is not supported in rsim mode. Use None or a CustomReferee instance.")

        return referee

    def data_update_listener(self, receiver: VisionReceiver):
        """Listener function to pull vision data from a VisionReceiver.

        This method is intended to be run in a separate thread and will call
        the receiver to continuously pull game data.

        Args:
            receiver: VisionReceiver instance to pull data from.
        """
        # Start receiving game data; this will run in a separate thread.
        receiver.pull_game_data()

    def start_threads(
        self,
        vision_receiver: VisionReceiver,
        referee_receiver: Optional[RefereeMessageReceiver] = None,
    ):
        """Start background threads for receiving vision and optionally referee data.

        Starts daemon threads so they do not prevent process exit.

        Args:
            vision_receiver: VisionReceiver to run in a background thread.
            referee_receiver: Optional RefereeMessageReceiver to run in a background thread.
        """
        # Give the receiver a handle to _stop_event so geometry errors can
        # signal the main loop to stop and re-raise the exception cleanly.
        vision_receiver._stop_event = self._stop_event
        self._vision_receiver = vision_receiver

        vision_thread = threading.Thread(target=vision_receiver.pull_game_data)

        vision_thread.daemon = True

        vision_thread.start()
        if referee_receiver is not None:
            referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
            referee_thread.daemon = True
            referee_thread.start()

    def _setup_sides_data(
        self,
        my_strategy: AbstractStrategy,
        opp_strategy: Optional[AbstractStrategy],
        filtering: bool,
        control_scheme: str,
        opp_control_scheme: Optional[str],
        my_trusted_ir_robots: Optional[FrozenSet[int]] = None,
        opp_trusted_ir_robots: Optional[FrozenSet[int]] = None,
        allowed_yellow_ids: Optional[FrozenSet[int]] = None,
        allowed_blue_ids: Optional[FrozenSet[int]] = None,
    ) -> Tuple[SideRuntime, Optional[SideRuntime]]:
        """Setup the data structures for both sides (my team and opponent)
        Args:
            my_strategy (AbstractStrategy): The strategy for the friendly team.
            opp_strategy (Optional[AbstractStrategy]): The strategy for the opponent team. If None, opponent side will be None.
            filtering (bool): Whether to use filtering in the position refiners.
            control_scheme (str): Name of the motion control scheme to use for the friendly team.
            opp_control_scheme (Optional[str]): Name of the motion control scheme to use for the opponent team. If not set, uses same as friendly.
            my_trusted_ir_robots (FrozenSet[int], optional): Vision IDs of friendly robots whose IR sensor is trusted.
            opp_trusted_ir_robots (FrozenSet[int], optional): Vision IDs of opponent robots whose IR sensor is trusted.
            allowed_yellow_ids (FrozenSet[int], optional): Vision IDs of yellow robots in play; others are ignored.
            allowed_blue_ids (FrozenSet[int], optional): Vision IDs of blue robots in play; others are ignored.

        Side effect: Initializes the SideRuntime for both friendly and opponent sides, including their strategies, refiners, and motion controllers.

        Returns:
            Tuple containing the SideRuntime for the friendly team and the opponent team (or None if no opponent strategy provided).
        """
        opp_side = None
        my_pos_ref, my_vel_ref, my_robot_ref = self._init_refiners(
            self.full_field_dims,
            filtering=filtering,
            exp_ball=self.exp_ball,
            trusted_ir_robots=my_trusted_ir_robots,
            allowed_yellow_ids=allowed_yellow_ids,
            allowed_blue_ids=allowed_blue_ids,
        )
        my_motion_controller = get_control_scheme(control_scheme)
        my_strategy.setup_strategy_blackboard(is_opp_strat=False)
        my_side = SideRuntime(
            strategy=my_strategy,
            position_refiner=my_pos_ref,
            velocity_refiner=my_vel_ref,
            robot_info_refiner=my_robot_ref,
            motion_controller=my_motion_controller,
        )

        if opp_strategy is not None:
            opp_pos_ref, opp_vel_ref, opp_robot_ref = self._init_refiners(
                self.full_field_dims,
                filtering=filtering,
                exp_ball=self.exp_ball,
                trusted_ir_robots=opp_trusted_ir_robots,
                allowed_yellow_ids=allowed_yellow_ids,
                allowed_blue_ids=allowed_blue_ids,
            )
            opp_motion_controller = (
                get_control_scheme(opp_control_scheme) if opp_control_scheme is not None else my_motion_controller
            )
            opp_strategy.setup_strategy_blackboard(is_opp_strat=True)
            opp_side = SideRuntime(
                strategy=opp_strategy,
                position_refiner=opp_pos_ref,
                velocity_refiner=opp_vel_ref,
                robot_info_refiner=opp_robot_ref,
                motion_controller=opp_motion_controller,
            )

        return my_side, opp_side

    def _split_robot_responses_by_team(
        self, responses: List[RobotResponse]
    ) -> Tuple[List[RobotResponse], List[RobotResponse]]:
        """Split a list of RobotResponse objects into separate lists for the friendly and opponent teams
        based on the robot IDs and the vision.
        """
        friendly_responses = []
        opponent_responses = []

        for response in responses:
            cmd_id = response.id

            if self.my_team_is_yellow:
                vision_id = self.yellow_cmd_to_vision_mapping.get(cmd_id)
                if vision_id is not None:
                    friendly_responses.append(RobotResponse(vision_id, response.has_ball))
                else:
                    opp_vision_id = self.blue_cmd_to_vision_mapping.get(cmd_id)
                    if opp_vision_id is not None:
                        opponent_responses.append(RobotResponse(opp_vision_id, response.has_ball))
                    else:
                        self.logger.warning(f"RobotResponse cmd_id={cmd_id} not found in either yellow or blue mapping")
            else:
                vision_id = self.blue_cmd_to_vision_mapping.get(cmd_id)
                if vision_id is not None:
                    friendly_responses.append(RobotResponse(vision_id, response.has_ball))
                else:
                    opp_vision_id = self.yellow_cmd_to_vision_mapping.get(cmd_id)
                    if opp_vision_id is not None:
                        opponent_responses.append(RobotResponse(opp_vision_id, response.has_ball))
                    else:
                        self.logger.warning(f"RobotResponse cmd_id={cmd_id} not found in either blue or yellow mapping")

        return friendly_responses, opponent_responses

    def _update_robot_feedback_snapshot(self, responses: List[RobotResponse], now: float) -> None:
        _record_robot_feedback_responses(self._robot_feedback_by_port_id, responses, now)
        self._robot_feedback_snapshot = _build_robot_feedback_snapshot(
            self._robot_feedback_by_port_id,
            now=now,
            my_team_is_yellow=self.my_team_is_yellow,
            yellow_cmd_to_vision_mapping=self.yellow_cmd_to_vision_mapping,
            blue_cmd_to_vision_mapping=self.blue_cmd_to_vision_mapping,
        )

    def _push_robot_feedback_to_referee(self) -> None:
        if isinstance(self.referee, CustomReferee):
            self.referee.set_robot_feedback_data(self._robot_feedback_snapshot)

    def _remove_rsim_ball(self):
        """Removes the ball from the RSim environment by teleporting it off-field."""
        self.sim_controller.remove_ball()
        self.rsim_env.step_noop()  # Step the environment to apply the change

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
        # No sim to load for real mode.
        if self.mode == Mode.REAL:
            return None, None

        get_formations_kwargs = dict(
            bounds=self.field_bounds,
            n_right=self.exp_friendly if self.my_team_is_right else self.exp_enemy,
            n_left=self.exp_enemy if self.my_team_is_right else self.exp_friendly,
        )
        if self.formation_type is not None:
            get_formations_kwargs["formation_type"] = self.formation_type
        right_start, left_start = get_formations(**get_formations_kwargs)

        yellow_start, blue_start = map_left_right_to_colors(
            self.my_team_is_yellow, self.my_team_is_right, right_start, left_start
        )

        if self.mode == Mode.RSIM:
            n_yellow, n_blue = map_friendly_enemy_to_colors(self.my_team_is_yellow, self.exp_friendly, self.exp_enemy)
            rsim_env = SSLStandardEnv(
                n_robots_yellow=n_yellow,
                n_robots_blue=n_blue,
                render_mode=None,
                blue_starting_formation=blue_start,
                yellow_starting_formation=yellow_start,
                full_field_dims=self.full_field_dims,
                ball_starting_position=self.field_bounds.center,
                gaussian_noise=rsim_noise,
                vanishing=rsim_vanishing,
            )

            if self.opp:
                self.opp.strategy.load_rsim_env(rsim_env)
            self.my.strategy.load_rsim_env(rsim_env)

            return rsim_env, RSimController(field_bounds=self.field_bounds, exp_ball=self.exp_ball, env=rsim_env)

        # GRSIM Mode
        else:
            # can consider baking all of these directly into sim controller
            sim_controller = GRSimController(self.field_bounds, self.exp_ball)
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
            for y in y_to_keep:
                sim_controller.set_robot_presence(y, True, True)
                y_start = yellow_start[y]
                sim_controller.teleport_robot(True, y, y_start[0], y_start[1], y_start[2])
            for b in b_to_keep:
                sim_controller.set_robot_presence(b, False, True)
                b_start = blue_start[b]
                sim_controller.teleport_robot(False, b, b_start[0], b_start[1], b_start[2])

            if self.exp_ball:
                sim_controller.teleport_ball(self.field_bounds.center[0], self.field_bounds.center[1])
            else:
                sim_controller.remove_ball()

            return None, sim_controller

    def _setup_vision_and_referee(self) -> Tuple[deque, deque]:
        """Setup vision and referee buffers, starting network receivers for gRSim/Real."""
        vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
        ref_buffer = deque(maxlen=1)
        if self.mode != Mode.RSIM:
            on_geometry = self._make_geometry_validation_callback()
            vision_receiver = VisionReceiver(vision_buffers, on_geometry=on_geometry)
            if isinstance(self.referee, OfficialReferee):
                self.start_threads(vision_receiver, RefereeMessageReceiver(ref_buffer))
            else:
                self.start_threads(vision_receiver)
        return vision_buffers, ref_buffer

    def _make_geometry_validation_callback(self):
        """Return a one-shot callback that validates vision geometry matches full_field_dims.

        Raises RuntimeError if the field size reported by gRSim/SSL-Vision differs from
        full_field_dims by more than _GEOMETRY_MATCH_TOLERANCE_M metres.
        """

        def _on_geometry(field_size) -> None:
            vision_half_length = field_size.field_length / 2000.0
            vision_half_width = field_size.field_width / 2000.0
            d = self.full_field_dims
            t = _GEOMETRY_MATCH_TOLERANCE_M
            mismatches = []
            # simplify the verification as the field dim packets can be non-standard
            if abs(vision_half_length - d.full_field_half_length) > t:
                mismatches.append(
                    f"field_length: vision={vision_half_length * 2:.3f}m configured={d.full_field_half_length * 2:.3f}m"
                )
            if abs(vision_half_width - d.full_field_half_width) > t:
                mismatches.append(
                    f"field_width: vision={vision_half_width * 2:.3f}m configured={d.full_field_half_width * 2:.3f}m"
                )
            if mismatches:
                raise RuntimeError(
                    "Field geometry mismatch between full_field_dims and vision packet:\n"
                    + "\n".join(f"  {m}" for m in mismatches)
                    + "\nUpdate full_field_dims in StrategyRunner to match the actual field."
                )

        return _on_geometry

    def _assert_exp_robots_and_ball(
        self,
        exp_friendly: int,
        exp_enemy: int,
        exp_ball: bool,
    ) -> None:
        """
        Validate that expected robot counts and ball presence are consistent
        with both team strategies at runtime.

        This method performs runtime configuration checks to ensure that the
        expected number of friendly and opponent robots, as well as ball
        availability, match the assumptions declared by each strategy.

        Parameters
        ----------
        exp_friendly : int
            Expected number of friendly robots.
        exp_enemy : int
            Expected number of opponent robots.
        exp_ball : bool
            Whether a ball is expected to be present.

        Raises
        ------
        ValueError
            If robot counts fall outside valid bounds.
        RuntimeError
            If strategy expectations do not match runtime configuration.
        """

        if exp_friendly > MAX_ROBOTS:
            raise ValueError(f"Expected number of friendly robots ({exp_friendly}) exceeds MAX_ROBOTS ({MAX_ROBOTS}).")
        if exp_enemy > MAX_ROBOTS:
            raise ValueError(f"Expected number of enemy robots ({exp_enemy}) exceeds MAX_ROBOTS ({MAX_ROBOTS}).")
        if exp_friendly < 1:
            raise ValueError(f"Expected number of friendly robots ({exp_friendly}) must be at least 1.")
        if exp_enemy < 0:
            raise ValueError(f"Expected number of enemy robots ({exp_enemy}) cannot be negative.")

        if not self.my.strategy.assert_exp_robots(exp_friendly, exp_enemy):
            raise RuntimeError("Runtime robot count does not match expectations of my strategy.")

        if not exp_ball and self.my.strategy.exp_ball:
            raise RuntimeError("Ball expected by my strategy, but not available in runtime configuration.")

        if self.opp:
            if not self.opp.strategy.assert_exp_robots(exp_enemy, exp_friendly):
                raise RuntimeError("Runtime robot count does not match expectations of opponent strategy.")

            if not exp_ball and self.opp.strategy.exp_ball:
                raise RuntimeError("Ball expected by opponent strategy, but not available in runtime configuration.")

    def _assert_exp_goals(self):
        """Assert the expected number of goals."""
        if not self.my.strategy.assert_exp_goals(
            self.my.game.field.includes_my_goal_line,
            self.my.game.field.includes_opp_goal_line,
        ):
            raise RuntimeError("Field does not match expected goals for my strategy.")
        if self.opp:
            if not self.opp.strategy.assert_exp_goals(
                self.opp.game.field.includes_my_goal_line,
                self.opp.game.field.includes_opp_goal_line,
            ):
                raise RuntimeError("Field does not match expected goals for opponent strategy.")

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
            my_viz_to_cmd_mapping, opp_viz_to_cmd_mapping = map_colors_to_friendly_enemy(
                self.my_team_is_yellow,
                self.yellow_vision_to_cmd_mapping,
                self.blue_vision_to_cmd_mapping,
            )
            my_robot_controller = RealRobotController(
                is_team_yellow=self.my_team_is_yellow,
                n_friendly=self.exp_friendly,
                vision_to_cmd_mapping=my_viz_to_cmd_mapping,
            )
            if self.opp:
                serial = my_robot_controller.serial_port  # share serial connection for efficiency
                opp_robot_controller = RealRobotController(
                    is_team_yellow=not self.my_team_is_yellow,
                    n_friendly=self.exp_enemy,
                    vision_to_cmd_mapping=opp_viz_to_cmd_mapping,
                    serial_port=serial,
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
        field_dims: FieldDimensions,
        filtering: bool,
        exp_ball: bool = True,
        trusted_ir_robots: Optional[FrozenSet[int]] = None,
        allowed_yellow_ids: Optional[FrozenSet[int]] = None,
        allowed_blue_ids: Optional[FrozenSet[int]] = None,
    ) -> tuple[PositionRefiner, VelocityRefiner, RobotInfoRefiner]:
        """
        Initialize the position, velocity, and robot info refiners.
        Args:
            field_dims (FieldDimensions): The field dimensions.
            filtering (bool): Whether to use filtering in the position refiner.
            exp_ball (bool): Whether the ball is expected. When False, the position refiner is
                             allowed to return None if no ball is detected in raw vision data.
            trusted_ir_robots (FrozenSet[int], optional): Vision IDs of robots whose IR sensor is trusted.
                See RobotInfoRefiner for details.
            allowed_yellow_ids (FrozenSet[int], optional): Vision IDs of yellow robots that are in play.
                Any robot ID seen by vision that is not in this set will be ignored.
            allowed_blue_ids (FrozenSet[int], optional): Vision IDs of blue robots that are in play.
                Any robot ID seen by vision that is not in this set will be ignored.
        Returns:
            tuple: The initialized PositionRefiner, VelocityRefiner, and RobotInfoRefiner.
        """
        position_refiner = PositionRefiner(
            field_dims,
            filtering=filtering,
            exp_ball=exp_ball,
            allowed_yellow_ids=allowed_yellow_ids,
            allowed_blue_ids=allowed_blue_ids,
        )
        velocity_refiner = VelocityRefiner()
        robot_info_refiner = RobotInfoRefiner(trusted_ir_robots=trusted_ir_robots)

        return position_refiner, velocity_refiner, robot_info_refiner

    def _load_game(self):
        """
        Load the game state for both friendly and opponent strategies after waiting for valid game data with GameGater.

        Side effect: Populates game, game_history and current_game_frame on self.my (and self.opp if present).
        """
        if self.mode == Mode.REAL:
            my_mapping = (
                self.yellow_vision_to_cmd_mapping if self.my_team_is_yellow else self.blue_vision_to_cmd_mapping
            )
            opp_mapping = (
                (self.blue_vision_to_cmd_mapping if self.my_team_is_yellow else self.yellow_vision_to_cmd_mapping)
                if self.opp
                else {}
            )
        else:
            my_mapping = None
            opp_mapping = None

        my_current_game_frame, opp_current_game_frame = GameGater.wait_until_game_valid(
            self.my_team_is_yellow,
            self.my_team_is_right,
            self.exp_friendly,
            self.exp_enemy,
            self.exp_ball,
            self.vision_buffers,
            self.my.position_refiner,
            is_pvp=self.opp is not None,
            rsim_env=self.rsim_env,
            my_vision_to_cmd_mapping=my_mapping,
            opp_vision_to_cmd_mapping=opp_mapping,
        )

        self.my.position_refiner.start_filtering()
        if self.opp:
            self.opp.position_refiner.start_filtering()

        my_field = Field(self.my_team_is_right, self.full_field_dims, self.field_bounds)
        self.my.game_history = GameHistory(MAX_GAME_HISTORY)
        self.my.game = Game(self.my.game_history, my_current_game_frame, field=my_field)
        self.my.current_game_frame = my_current_game_frame

        if self.opp:
            opp_field = Field(not self.my_team_is_right, self.full_field_dims, self.field_bounds)
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

    def _stop_robots(self, repeat: int = 1):
        """
        Send stop commands to the robots.
        Args:
            repeat (int): Number of times to send the stop command.
        """

        def build_commands(team: SideRuntime) -> dict[int, RobotCommand]:
            return {robot_id: RobotCommand(0, 0, 0, 0, 0, 0) for robot_id in team.game.friendly_robots.keys()}

        my_cmds = build_commands(self.my) if self.my.game is not None else None
        opp_cmds = build_commands(self.opp) if self.opp and self.opp.game is not None else None

        for _ in range(repeat):
            if my_cmds:
                self.my.strategy.robot_controller.add_robot_commands(my_cmds)
                self.my.strategy.robot_controller.send_robot_commands()

            if opp_cmds:
                self.opp.strategy.robot_controller.add_robot_commands(opp_cmds)
                self.opp.strategy.robot_controller.send_robot_commands()

    def close(self, stop_command_repeat: int = 20):
        """
        Close resources used by the StrategyRunner and stop robots if in real mode.
        Args:
            stop_command_repeat (int): Number of times to send the stop command to robots.
        """
        self.logger.info("Cleaning up resources...")

        if self.mode == Mode.REAL:
            try:
                self._stop_robots(repeat=stop_command_repeat)
            except Exception:
                self.logger.exception("Was unable to stop robots cleanly.")
        if self.profiler:
            self.profiler.disable()
            if self.profiler.getstats():
                self.profiler.dump_stats(f"{self.profiler_name}.prof")
        if self.replay_writer:
            self.replay_writer.close()
        if self.vision_stream:
            self.vision_stream.stop()
        if self.rsim_env:
            self.rsim_env.close()
        self._stop_fps_live()

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
                        self._stop_robots()
                        break
                    elif status == TestingStatus.SUCCESS:
                        self._stop_robots()
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
        continues until interrupted via SIGINT stop event, after which resources
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
            # Loop exited via stop_event — check whether a background thread
            # parked an exception (e.g. geometry mismatch from VisionReceiver).
            if self._vision_receiver is not None and self._vision_receiver.thread_exception is not None:
                raise self._vision_receiver.thread_exception
        except Exception:
            if self._stop_event.is_set() and (
                self._vision_receiver is None or self._vision_receiver.thread_exception is None
            ):
                self.logger.info("Stopping run loop due to interrupt.")
            else:
                self.logger.exception("Exception occurred during run loop:")
                raise
        finally:
            self.close()

    def step_once(self):
        """Advance the runner by one strategy/simulation tick.

        This is a public wrapper around the internal single-step loop for
        deterministic harnesses and scenario runners that need to apply events
        or assertions between ticks without taking ownership of the runner's
        game-loop internals.
        """
        self._run_step()

    def _run_step(self):
        """Perform one tick of the overall game loop.

        This collects vision frames, alternates which side runs first, steps
        the strategies, writes replay frames if enabled, and enforces timestep
        rate-limiting (sleeping when necessary).

        No return value; updates internal game state and controllers.
        """
        frame_start = time.perf_counter()

        # Re-raise any exception parked by a background thread (e.g. geometry
        # mismatch from VisionReceiver) so the main loop exits with a clean
        # traceback instead of crashing somewhere unrelated later.
        if self._vision_receiver is not None and self._vision_receiver.thread_exception is not None:
            raise self._vision_receiver.thread_exception
        self._draw_rsim_field_bounds_overlay()

        raw_robot_responses: List[RobotResponse] = []
        if self.mode == Mode.REAL:
            raw_robot_responses = self.my.strategy.robot_controller.get_robots_responses() or []
            self._update_robot_feedback_snapshot(raw_robot_responses, frame_start)
            self._push_robot_feedback_to_referee()

        if isinstance(self.referee, CustomReferee):
            ref_data = self.referee.step(self.my.current_game_frame, self.my.current_game_frame.ts)
            self.ref_buffer.append(ref_data)
            _BALL_PLACEMENT_COMMANDS = (
                RefereeCommand.BALL_PLACEMENT_YELLOW,
                RefereeCommand.BALL_PLACEMENT_BLUE,
            )
            if self.sim_controller is not None and ref_data.designated_position is not None:
                if (
                    ref_data.referee_command == RefereeCommand.STOP
                    and self._prev_custom_ref_command != RefereeCommand.STOP
                ):
                    # On transition into STOP with a designated position, teleport
                    # the ball immediately and skip straight to FORCE_START so
                    # simulation doesn't wait for physical ball placement.
                    x, y = ref_data.designated_position
                    self.sim_controller.teleport_ball(x, y)
                    self.referee.force_command(RefereeCommand.FORCE_START, self.my.current_game_frame.ts)
                elif (
                    ref_data.referee_command in _BALL_PLACEMENT_COMMANDS
                    and self._prev_custom_ref_command not in _BALL_PLACEMENT_COMMANDS
                ):
                    # On transition into BALL_PLACEMENT, teleport the ball to the
                    # designated position and let the state machine auto-advance.
                    # Robots cannot physically retrieve an out-of-bounds ball in
                    # simulation, so we simulate placement instantly.
                    x, y = ref_data.designated_position
                    self.sim_controller.teleport_ball(x, y)
            self._prev_custom_ref_command = ref_data.referee_command

        if self.mode == Mode.RSIM:
            obs = self.rsim_env._frame_to_observations()
            vision_frames = [obs[0]]
            referee_data = self.ref_buffer.popleft() if self.ref_buffer else None
        else:
            vision_frames = [buffer.popleft() if buffer else None for buffer in self.vision_buffers]
            if self.ref_buffer:
                self._last_referee_data = self.ref_buffer.popleft()
            referee_data = self._last_referee_data

        friendly_res, opp_res = None, None
        if self.mode == Mode.REAL:
            responses = raw_robot_responses
            if self.opp:
                friendly_res, opp_res = self._split_robot_responses_by_team(responses)
            else:
                cmd_to_vision = (
                    self.yellow_cmd_to_vision_mapping if self.my_team_is_yellow else self.blue_cmd_to_vision_mapping
                )
                if cmd_to_vision:
                    friendly_res = []
                    for r in responses:
                        vision_id = cmd_to_vision.get(r.id)
                        if vision_id is None:
                            self.logger.warning(f"RobotResponse cmd_id={r.id} not found in mapping for controlled team")
                            continue
                        friendly_res.append(RobotResponse(vision_id, r.has_ball))
                else:
                    friendly_res = responses

        # alternate between opp and friendly playing
        real = self.mode == Mode.REAL
        if self.toggle_opp_first:
            if self.opp:
                self._step_game(
                    vision_frames,
                    referee_data,
                    True,
                    real_responses=opp_res if real else None,
                )
            self._step_game(
                vision_frames,
                referee_data,
                False,
                real_responses=friendly_res if real else None,
            )
        else:
            self._step_game(
                vision_frames,
                referee_data,
                False,
                real_responses=friendly_res if real else None,
            )
            if self.opp:
                self._step_game(
                    vision_frames,
                    referee_data,
                    True,
                    real_responses=opp_res if real else None,
                )
        self.toggle_opp_first = not self.toggle_opp_first
        self._publish_vision_stream_frame()
        self._push_bt_nodes_to_referee()

        # --- rate limiting ---
        if self.mode != Mode.RSIM:
            processing_time = time.perf_counter() - frame_start
            wait_time = max(0, TIMESTEP - processing_time)
            time.sleep(wait_time)

        # --- end of frame ---
        if self.show_live_status:
            frame_end = time.perf_counter()
            frame_dt = frame_end - frame_start

            self.elapsed_time += frame_dt
            self.num_frames_elapsed += 1

            if self.elapsed_time >= FPS_PRINT_INTERVAL:
                fps = self.num_frames_elapsed / self.elapsed_time

                ref = self.referee_refiner
                stage_secs = ref.stage_time_left
                stage_min = int(stage_secs // 60)
                stage_sec = int(stage_secs % 60)
                display = Text()
                display.append(f"FPS: {fps:.1f}", style="bold cyan")
                display.append("  |  ")
                display.append(ref.last_command.name, style="bold yellow")
                if ref.last_next_command:
                    display.append("  ->  ")
                    display.append(ref.last_next_command.name, style="yellow")
                display.append("  |  ")
                display.append(ref.stage.name.replace("_", " ").title())
                display.append("  |  Blue ")
                display.append(str(ref.blue_team.score), style="bold blue")
                display.append(" - ")
                display.append(str(ref.yellow_team.score), style="bold yellow")
                display.append(" Yellow")
                display.append(f"  |  {stage_min}:{stage_sec:02d} left")
                display.append("  |  Ref: ")
                if isinstance(self.referee, CustomReferee):
                    display.append("custom", style="bold magenta")
                    display.append(" (")
                    display.append(self.referee.profile_name, style="magenta")
                    display.append(")")
                elif isinstance(self.referee, OfficialReferee):
                    display.append("official", style="bold magenta")
                else:
                    display.append("none", style="bold magenta")

                if ref.last_status_message:
                    display.append(f"  |  {ref.last_status_message}", style="dim")

                self._fps_live.update(display)
                self._fps_live.refresh()

                self.elapsed_time = 0.0
                self.num_frames_elapsed = 0

    def _publish_vision_stream_frame(self) -> None:
        """Publish the latest refined game frame to the browser stream."""
        if self.vision_stream is None or self._vision_stream_renderer is None or self.my.current_game_frame is None:
            return
        if not self.vision_stream.is_due():
            return
        self.vision_stream.publish_status(self._vision_stream_status())
        frame = self._vision_stream_renderer.render(self.my.current_game_frame)
        self.vision_stream.publish_rgb_frame(frame)

    _VS_TEAM_NAMES = [
        "FC Recursion",
        "Real Segfault",
        "Borussia Debugmund",
        "Manchester Bytecode",
        "Inter Malloc",
        "Atletico del Stack",
        "Null Pointer United",
        "Schalke 0x04",
        "Deportivo Kernel",
        "Racing Club de Runtime",
        "Galactic Overhead",
        "Ajax Exception",
        "Off-by-One City",
        "SV Deadlock",
        "Infinite Loop FC",
    ]

    _VS_IDLE_LINES = [
        "The robots are thinking...",
        "Calculating optimal trajectory",
        "Both sides plotting their next move",
        "The crowd holds its breath",
        "Pure silicon determination out there",
        "No human reflexes required",
        "Running at full clock speed",
        "Algorithms at war",
        "404: defence not found",
        "This is peak robot football",
    ]

    _VS_KICK_LINES = [
        "What a strike!",
        "They've let it fly!",
        "Big boot from the robot!",
        "The ball is moving!",
        "Powerful kick!",
        "Sending it downfield!",
    ]

    _VS_GOAL_LINES_YELLOW = [
        "GOAL! Yellow draws blood!",
        "Yellow scores! Unbelievable!",
        "The yellow machine delivers!",
        "Yellow puts it in the net!",
    ]

    _VS_GOAL_LINES_BLUE = [
        "GOAL! Blue strikes back!",
        "Blue finds the net!",
        "Brilliant from the blue side!",
        "Blue pulls one back!",
    ]

    _VS_FOOTBALLER_NAMES = [
        "Martin",
        "Fred",
        "Joel",
        "Louis",
    ]

    @staticmethod
    def _assign_team_names() -> tuple[str, str]:
        import random

        pool = list(StrategyRunner._VS_TEAM_NAMES)
        random.shuffle(pool)
        return pool[0], pool[1]

    def _update_vs_commentary(self, ball_speed: float | None, score_blue: int, score_yellow: int) -> None:
        import random

        now = time.monotonic()
        score = (score_blue, score_yellow)

        # Goal scored — highest priority
        if score != self._vs_prev_score:
            if score[1] > self._vs_prev_score[1]:
                line = random.choice(self._VS_GOAL_LINES_YELLOW)
            else:
                line = random.choice(self._VS_GOAL_LINES_BLUE)
            self._vs_commentary = line
            self._vs_commentary_until = now + 5.0
            self._vs_prev_score = score
            self._vs_prev_ball_speed = ball_speed or 0.0
            return

        self._vs_prev_score = score

        # Kick detected
        prev_spd = self._vs_prev_ball_speed
        cur_spd = ball_speed or 0.0
        self._vs_prev_ball_speed = cur_spd
        if cur_spd > _VS_KICK_THRESHOLD and prev_spd <= _VS_KICK_THRESHOLD:
            if now >= self._vs_commentary_until:
                self._vs_commentary = random.choice(self._VS_KICK_LINES)
                self._vs_commentary_until = now + 2.5

        # Idle rotation
        if now >= self._vs_idle_next:
            if now >= self._vs_commentary_until:
                self._vs_commentary = self._VS_IDLE_LINES[self._vs_idle_index % len(self._VS_IDLE_LINES)]
                self._vs_idle_index += 1
            self._vs_idle_next = now + 6.0

    def _vision_stream_status(self) -> dict[str, object]:
        """Build status metadata shown above the browser stream."""
        stage_secs = max(0.0, self.referee_refiner.stage_time_left)
        stage_min = int(stage_secs // 60)
        stage_sec = int(stage_secs % 60)

        blue = self.referee_refiner.blue_team
        yellow = self.referee_refiner.yellow_team

        ball = self.my.current_game_frame.ball if self.my.current_game_frame else None
        ball_speed = (ball.v.x**2 + ball.v.y**2) ** 0.5 if ball is not None else None

        name_yellow, name_blue = (
            self._vs_team_names if self.my_team_is_yellow else (self._vs_team_names[1], self._vs_team_names[0])
        )
        self._update_vs_commentary(ball_speed, blue.score, yellow.score)

        return {
            "time_left": f"{stage_min}:{stage_sec:02d}",
            "score_blue": blue.score,
            "score_yellow": yellow.score,
            "yellow_cards_blue": blue.yellow_cards,
            "yellow_cards_yellow": yellow.yellow_cards,
            "red_cards_blue": blue.red_cards,
            "red_cards_yellow": yellow.red_cards,
            "team_blue": name_blue,
            "team_yellow": name_yellow,
            "mode": self.mode.value,
            "ball_speed": round(ball_speed, 2) if ball_speed is not None else None,
            "commentary": self._vs_commentary,
            "annotations": self._vision_stream_annotations(),
            "roster": self._vision_stream_roster(),
        }

    def _get_robot_name(self, is_friendly: bool, robot_id: int) -> str:
        key = (is_friendly, robot_id)
        if key not in self._vs_robot_names:
            used = set(self._vs_robot_names.values())
            pool = [n for n in self._VS_FOOTBALLER_NAMES if n not in used]
            if not pool:
                pool = self._VS_FOOTBALLER_NAMES
            import random

            self._vs_robot_names[key] = random.choice(pool)
        return self._vs_robot_names[key]

    def _vision_stream_annotations(self) -> list[dict]:
        """Build per-robot label annotations (footballer name) for the overlay canvas."""
        renderer = self._vision_stream_renderer
        game_frame = self.my.current_game_frame
        if renderer is None or game_frame is None:
            return []

        annotations = []
        for robot in game_frame.friendly_robots.values():
            name = self._get_robot_name(True, robot.id)
            px, py = renderer._pos_transform(robot.p.x, -robot.p.y)
            annotations.append({"id": robot.id, "team": "friendly", "px": px, "py": py, "label": name})

        for robot in game_frame.enemy_robots.values():
            name = self._get_robot_name(False, robot.id)
            px, py = renderer._pos_transform(robot.p.x, -robot.p.y)
            annotations.append({"id": robot.id, "team": "enemy", "px": px, "py": py, "label": name})

        return annotations

    def _push_bt_nodes_to_referee(self) -> None:
        """Extract per-robot debug status and push to CustomReferee for GUI display.

        Strategies that aren't behaviour-tree-based (e.g. `KernelStrategy`)
        have no `RUNNING` BT nodes to walk. Any strategy may instead expose a
        `debug_status() -> dict[int, list[str]]` method to report its own
        equivalent of "what is this robot's tactic doing right now" — used
        in preference to the BT walk below when present, so the same GUI
        panel works for both without either strategy family needing to know
        about the other.
        """
        if not isinstance(self.referee, CustomReferee):
            return
        if hasattr(self.my.strategy, "debug_status"):
            self.referee.set_bt_data(self.my.strategy.debug_status())
            return
        bt_nodes: dict[int, list[str]] = {}
        # Build node lookup and parent map
        node_by_id = {}
        parent_of = {}
        for n in self.my.strategy.behaviour_tree.root.iterate():
            node_by_id[n.id] = n
            if hasattr(n, "children"):
                for child in n.children:
                    parent_of[child.id] = n.id
        # Find RUNNING leaf nodes, walk up to root to collect path and robot_id.
        # robot_id may live on any ancestor (not just the leaf), so we scan the
        # full path rather than stopping at the leaf node.
        for node_id, status in self._bt_snapshot.visited.items():
            if status.name != "RUNNING":
                continue
            node = node_by_id.get(node_id)
            if node is None:
                continue
            # Walk from leaf to root: collect path names and search for robot_id.
            path = []
            rid = None
            cur = node_id
            while cur is not None:
                n = node_by_id.get(cur)
                if n is None:
                    break
                path.append(n.name)
                if rid is None:
                    if hasattr(n, "debug_state"):
                        state = n.debug_state()
                        if state and "robot_id" in state:
                            rid = state["robot_id"]
                    if rid is None and hasattr(n, "robot_id_key"):
                        try:
                            rid = n.blackboard.get(n.robot_id_key)
                        except Exception:
                            pass
                cur = parent_of.get(cur)
            if rid is not None:
                path.reverse()
                bt_nodes.setdefault(rid, []).append(" › ".join(path))
        self.referee.set_bt_data(bt_nodes)

    def _vision_stream_roster(self) -> list[dict]:
        """Build the player roster list shown below the scoreboard."""
        game_frame = self.my.current_game_frame
        if game_frame is None:
            return []

        _ROLE_LABELS = {
            "GOALKEEPER": "Goalkeeper",
            "DEFENDER": "Defender",
            "STRIKER": "Striker",
            "MIDFIELDER": "Midfielder",
            "UNASSIGNED": "On field",
        }

        role_map: dict = {}
        passer_id: int | None = None
        receiver_id: int | None = None
        try:
            bb = self.my.strategy.blackboard
            if bb is not None:
                role_map = bb.role_map or {}
                passer_id = bb.get("passer_id")
                receiver_id = bb.get("receiver_id")
        except Exception:
            pass

        roster = []
        for robot in game_frame.friendly_robots.values():
            name = self._get_robot_name(True, robot.id)
            if robot.id == passer_id:
                status = "Passing"
            elif robot.id == receiver_id:
                status = "Receiving"
            else:
                role = role_map.get(robot.id)
                status = _ROLE_LABELS.get(role.name, "On field") if role else "On field"
            roster.append(
                {"name": name, "team": "yellow" if game_frame.my_team_is_yellow else "blue", "status": status}
            )

        for robot in game_frame.enemy_robots.values():
            name = self._get_robot_name(False, robot.id)
            role = role_map.get(robot.id)
            status = _ROLE_LABELS.get(role.name, "On field") if role else "On field"
            roster.append(
                {"name": name, "team": "blue" if game_frame.my_team_is_yellow else "yellow", "status": status}
            )

        return roster

    def _draw_rsim_field_bounds_overlay(self) -> None:
        """Draw active field bounds overlay in RSIM human render mode."""
        if self.mode != Mode.RSIM or not self.rsim_env:
            return

        # Overlays are cleared during render; only enqueue when the frame will be rendered.
        if self.rsim_env.render_mode != "human":
            return

        top_left = self.field_bounds.top_left
        bottom_right = self.field_bounds.bottom_right

        bounds_polygon = [
            (top_left[0], top_left[1]),
            (bottom_right[0], top_left[1]),
            (bottom_right[0], bottom_right[1]),
            (top_left[0], bottom_right[1]),
        ]
        self.rsim_env.draw_polygon(bounds_polygon, color="PINK", width=2)

    def _step_game(
        self,
        vision_frames: List[RawVisionData],
        referee_data,
        running_opp: bool,
        real_responses: Optional[List[RobotResponse]] = None,
    ):
        """Step the game for the robot controller and strategy.

        Args:
            vision_frames (List[RawVisionData]): The vision frames.
            referee_data: The referee data from RSim or network receiver.
            running_opp (bool): Whether to run the opponent strategy.
            real_responses (Optional[List[RobotResponse]]): The robot responses pulled for real.
                                                            We use a shared transmitter, so it cannot be pulled per side.
        """
        side = self.opp if running_opp else self.my

        # Pull responses from robot controller
        if self.mode != Mode.REAL:
            responses = side.strategy.robot_controller.get_robots_responses()
        else:
            responses = real_responses if real_responses is not None else []

        # Update game frame with refined information
        new_game_frame = side.position_refiner.refine(side.current_game_frame, vision_frames)
        new_game_frame = side.velocity_refiner.refine(side.game_history, new_game_frame)  # , robot_frame.imu_data)
        new_game_frame = side.robot_info_refiner.refine(new_game_frame, responses)
        new_game_frame = self.referee_refiner.refine(new_game_frame, referee_data)

        # Store updated game frame
        side.current_game_frame = new_game_frame

        # write to replay
        if self.replay_writer and (running_opp != self.replay_writer.replay_configs.is_my_perspective):
            self.replay_writer.write_frame(new_game_frame)

        side.game.add_game_frame(new_game_frame)
        side.strategy.step()
