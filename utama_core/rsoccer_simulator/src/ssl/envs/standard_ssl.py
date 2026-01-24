import logging
import math
import random
from typing import List, Tuple

from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.config.robot_params import RSIM_PARAMS
from utama_core.config.settings import (
    MAX_BALL_SPEED,
    MIN_RELEASE_SPEED,
    RELEASE_GAIN,
    TIMESTEP,
)
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.global_utils.math_utils import deg_to_rad, rad_to_deg
from utama_core.rsoccer_simulator.src.Entities import Ball, Frame, Robot
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.rsoccer_simulator.src.Utils import KDTree

logger = logging.getLogger(__name__)


class SSLStandardEnv(SSLBaseEnv):
    """
    Description:
        Environment stripped to be a lightweight simulator for testing and development.
    args:
        field_type
        Num
        0       Divison A pitch
        1       Division B pitch
        2       HW Challenge

        blue/yellow_starting_formation
        Type: List[Tuple[float, float, float]]
        Description:
            list of (x, y, theta) coords for each robot to spawn in (in meters and radians).
            See the default BLUE_START_ONE/YELLOW_START_ONE for reference.
    Observation:
        Type: Tuple[FrameData, List[RobotInfo], List[RobotInfo]]
        Num     Item
        0       contains position info of ball and robots on the field
        1       contains RobotInfo data (robot.has_ball) for yellow_robots
        2       contains RobotInfo data (robot.has_ball) for blue_robots

    Actions:
        Type: Box(5, )
        Num     Action
        0       id 0 Blue Global X Direction Speed (max set by self.max_v)
        1       id 0 Blue Global Y Direction Speed
        2       id 0 Blue Angular Speed (max set by self.max_w)
        3       id 0 Blue Kick x Speed (max set by self.kick_speed_x)
        4       id 0 Blue Dribbler (true if positive)
    """

    def __init__(
        self,
        field_type: int = 1,
        render_mode: str = "human",
        n_robots_blue: int = 6,
        n_robots_yellow: int = 6,
        time_step: float = TIMESTEP,
        blue_starting_formation: list[tuple] = None,
        yellow_starting_formation: list[tuple] = None,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step=time_step,
            render_mode=render_mode,
        )
        # Note: observation_space and action_space removed - not needed for non-RL use

        # set starting formation style for
        self.blue_formation = LEFT_START_ONE if not blue_starting_formation else blue_starting_formation
        self.yellow_formation = RIGHT_START_ONE if not yellow_starting_formation else yellow_starting_formation

        # Track dribbler state across steps so we can model ball release when
        # the dribbler turns off in a way that depends on robot speed.
        self.prev_dribbler_blue = [False] * self.n_robots_blue
        self.prev_dribbler_yellow = [False] * self.n_robots_yellow
        self.prev_forward_blue = [0.0] * self.n_robots_blue
        self.prev_forward_yellow = [0.0] * self.n_robots_yellow

        # Kick persistence state: List of (kick_speed, frames_remaining)
        self.kick_persist_blue = [(0.0, 0) for _ in range(self.n_robots_blue)]
        self.kick_persist_yellow = [(0.0, 0) for _ in range(self.n_robots_yellow)]

        # Saving latest observation for the step (step, observation)
        self.latest_observation = (-1, None)

        logger.info(f"{n_robots_blue}v{n_robots_yellow} SSL Environment Initialized")

    def reset(self, *, seed=None, options=None):
        self.reward_shaping_total = None
        # Reset dribbler tracking state
        for i in range(self.n_robots_blue):
            self.prev_dribbler_blue[i] = False
            self.prev_forward_blue[i] = 0.0
            self.kick_persist_blue[i] = (0.0, 0)
        for i in range(self.n_robots_yellow):
            self.prev_dribbler_yellow[i] = False
            self.prev_forward_yellow[i] = 0.0
            self.kick_persist_yellow[i] = (0.0, 0)

        self.latest_observation = (-1, None)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """
        Advance the simulation by one step while modelling ball release when the
        dribbler turns off. The ball's release speed is proportional to the
        robot's current speed and capped to a realistic maximum.
        """
        # Increment step counter and build low-level simulator commands
        self.steps += 1
        commands = self._get_commands(action)

        # Apply any implicit kicks caused by dribbler state transitions.
        self._apply_dribbler_release_kicks(commands)

        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator and update dribbler/speed history
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()
        self._update_dribbler_history(commands)

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        if self.render_mode == "human":
            self.render()

        # Expose reward shaping totals via the info dict (for compatibility
        # with existing callers of SSLStandardEnv)
        return observation, reward, done, False, self.reward_shaping_total

    def _frame_to_observations(
        self,
    ) -> Tuple[RawVisionData, RobotResponse, RobotResponse]:
        """Return observation data that aligns with grSim.

        Returns (vision_observation, yellow_robot_feedback, blue_robot_feedback)
        vision_observation: closely aligned to SSLVision that returns a FramData object
        yellow_robots_info: feedback from individual yellow robots that returns a List[RobotInfo]
        blue_robots_info: feedback from individual blue robots that returns a List[RobotInfo]
        """
        if self.latest_observation[0] == self.steps:
            return self.latest_observation[1]

        # Ball observation shared by all robots
        ball_obs = RawBallData(self.frame.ball.x, -self.frame.ball.y, self.frame.ball.z, 1.0)

        # Robots observation (Blue + Yellow)
        blue_obs = []
        blue_robots_info = []
        for i in range(len(self.frame.robots_blue)):
            robot = self.frame.robots_blue[i]
            robot_pos, robot_info = self._get_robot_observation(robot)
            blue_obs.append(robot_pos)
            blue_robots_info.append(robot_info)

        yellow_obs = []
        yellow_robots_info = []
        for i in range(len(self.frame.robots_yellow)):
            robot = self.frame.robots_yellow[i]
            robot_pos, robot_info = self._get_robot_observation(robot)
            yellow_obs.append(robot_pos)
            yellow_robots_info.append(robot_info)

        # Return the complete shared observation
        # note that ball_obs stored in list to standardise with SSLVision
        # As there is sometimes multiple possible positions for the ball

        # Camera id as 0, only one camera for RSim
        result = (
            RawVisionData(self.time_step * self.steps, yellow_obs, blue_obs, [ball_obs], 0),
            yellow_robots_info,
            blue_robots_info,
        )
        self.latest_observation = (self.steps, result)
        return result

    def _get_robot_observation(self, robot):
        robot_pos = RawRobotData(robot.id, robot.x, -robot.y, -float(deg_to_rad(robot.theta)), 1)
        robot_info = RobotResponse(robot.id, robot.infrared)
        return robot_pos, robot_info

    def _get_commands(self, actions) -> list[Robot]:
        commands = []

        # Blue robots
        for i in range(self.n_robots_blue):
            v_x = actions["team_blue"][i][0]
            v_y = actions["team_blue"][i][1]
            v_theta = actions["team_blue"][i][2]

            dribbler = actions["team_blue"][i][4] > 0
            kick_v_x = RSIM_PARAMS.KICK_SPD if actions["team_blue"][i][3] > 0 else 0.0

            cmd = Robot(
                yellow=False,  # Blue team
                id=i,  # ID of the robot
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_v_x,
                dribbler=dribbler,
            )
            commands.append(cmd)

        # Yellow robots
        for i in range(self.n_robots_yellow):
            v_x = actions["team_yellow"][i][0]
            v_y = actions["team_yellow"][i][1]
            v_theta = actions["team_yellow"][i][2]

            dribbler = actions["team_yellow"][i][4] > 0
            kick_v_x = RSIM_PARAMS.KICK_SPD if actions["team_yellow"][i][3] > 0 else 0.0

            cmd = Robot(
                yellow=True,  # Yellow team
                id=i,  # ID of the robot
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_v_x,
                dribbler=dribbler,
            )
            commands.append(cmd)

        return commands

    def _apply_dribbler_release_kicks(self, commands: list[Robot]) -> None:
        """Approximate kicks when dribblers turn off for robots that had possession.

        This mutates the provided ``commands`` list in-place, potentially
        increasing ``kick_v_x`` for robots whose dribbler transitioned from
        on to off while they were moving with the ball.
        """
        KICK_PERSISTENCE_FRAMES = 3

        n_blue = self.n_robots_blue
        for i in range(n_blue):
            # Update persistence
            spd, frames = self.kick_persist_blue[i]
            if frames > 0:
                self.kick_persist_blue[i] = (spd, frames - 1)
                commands[i].kick_v_x = max(commands[i].kick_v_x, spd)

            release = self._dribbler_release_kick(
                self.prev_dribbler_blue,
                self.prev_forward_blue,
                i,
                commands[i].dribbler,
            )
            if release > 0.0:
                # Trigger new persistence
                self.kick_persist_blue[i] = (release, KICK_PERSISTENCE_FRAMES)
                commands[i].kick_v_x = max(commands[i].kick_v_x, release)

        for j in range(self.n_robots_yellow):
            cmd_idx = n_blue + j
            # Update persistence
            spd, frames = self.kick_persist_yellow[j]
            if frames > 0:
                self.kick_persist_yellow[j] = (spd, frames - 1)
                commands[cmd_idx].kick_v_x = max(commands[cmd_idx].kick_v_x, spd)

            release = self._dribbler_release_kick(
                self.prev_dribbler_yellow,
                self.prev_forward_yellow,
                j,
                commands[cmd_idx].dribbler,
            )
            if release > 0.0:
                self.kick_persist_yellow[j] = (release, KICK_PERSISTENCE_FRAMES)
                commands[cmd_idx].kick_v_x = max(commands[cmd_idx].kick_v_x, release)

    def _update_dribbler_history(self, commands: list[Robot]) -> None:
        """Update previous dribbler and forward velocity history for all robots."""
        n_blue = self.n_robots_blue
        # Update in-place to avoid list allocation
        for i in range(n_blue):
            self.prev_dribbler_blue[i] = commands[i].dribbler
            self.prev_forward_blue[i] = commands[i].v_x
        for j in range(self.n_robots_yellow):
            self.prev_dribbler_yellow[j] = commands[n_blue + j].dribbler
            self.prev_forward_yellow[j] = commands[n_blue + j].v_x

    def _dribbler_release_kick(
        self,
        prev_dribbler: List[bool],
        prev_forward: List[float],
        index: int,
        dribbler: bool,
    ) -> float:
        """Estimate the kick needed to release the ball when the dribbler turns off."""
        if not prev_dribbler[index] or dribbler:
            return 0.0

        # Require forward motion relative to heading to avoid releasing while backing up
        forward = prev_forward[index]
        if forward < MIN_RELEASE_SPEED:
            return 0.0

        return min(RELEASE_GAIN * forward, MAX_BALL_SPEED)

    def _calculate_reward_and_done(self):
        return 1, False

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame (random placement)"""
        pos_frame: Frame = Frame()

        for i in range(self.n_robots_blue):
            x, y, heading = self.blue_formation[i]
            pos_frame.robots_blue[i] = Robot(id=i, x=x, y=-y, theta=-rad_to_deg(heading))

        for i in range(self.n_robots_yellow):
            x, y, heading = self.yellow_formation[i]
            pos_frame.robots_yellow[i] = Robot(id=i, x=x, y=-y, theta=-rad_to_deg(heading))

        pos_frame.ball = Ball(x=0, y=0)

        return pos_frame

    def _get_random_position_frame(self):
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2

        def x(is_yellow=False):
            if is_yellow:
                return random.uniform(-half_len + 0.1, -0.2)
            else:
                return random.uniform(0.2, half_len - 0.1)

        def y():
            return random.uniform(-half_wid + 0.1, half_wid - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        def in_gk_area(obj):
            return obj.x > half_len - pen_len and abs(obj.y) < half_pen_wid

        pos_frame.ball = Ball(x=x(), y=y())
        while in_gk_area(pos_frame.ball):
            pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (x(False), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(False), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(id=i, x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(True), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(True), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(id=i, x=pos[0], y=pos[1], theta=theta())

        return pos_frame
