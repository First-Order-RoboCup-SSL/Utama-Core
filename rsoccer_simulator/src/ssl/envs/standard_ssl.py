import math
import random
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from rsoccer_simulator.src.Entities import Frame, Robot, Ball
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_simulator.src.Utils import KDTree
from team_controller.src.config.starting_formation import (
    BLUE_START_ONE,
    YELLOW_START_ONE,
)
from global_utils.math_utils import deg_to_rad, rad_to_deg

from entities.data.vision import BallData, RobotData, FrameData
from entities.data.command import RobotInfo


class SSLStandardEnv(SSLBaseEnv):
    """

    args:
        field_type
        Num
        0       Divison A pitch
        1       Division B pitch
        2       HW Challenge

        blue/yellow_starting_formation
        Type: List[Tuple[float, float, float]]
        Description:
            list of (x, y, theta) coords for each robot to spawn in.
            See the default BLUE_START_ONE/YELLOW_START_ONE for reference.

    Description:
        Environment stripped to be a lightweight simulator for testing and development.
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

    Reward:
        +5 if goal (blue)
    Starting State:
        Robots on thier respective sides, ball and defenders randomly positioned on
        positive field side.
    Episode Termination:
        Goal, 25 seconds (1000 steps), or rule infraction
    """

    def __init__(
        self,
        field_type: int = 1,
        render_mode: str = "human",
        n_robots_blue: int = 6,
        n_robots_yellow: int = 6,
        time_step: float = 0.0167,
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
        # Shared observation space for all robots:
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(4 + (self.n_robots_blue + self.n_robots_yellow) * 8,),
            dtype=np.float32,
        )

        # Action space for one robot:
        robot_action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Team action space:
        # Define action space for 6 robots in both teams
        self.action_space = gym.spaces.Dict(
            {
                "team_blue": gym.spaces.Tuple(
                    [robot_action_space for _ in range(self.n_robots_blue)]
                ),
                "team_yellow": gym.spaces.Tuple(
                    [robot_action_space for _ in range(self.n_robots_yellow)]
                ),
            }
        )

        # Set scales for rewards
        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length / 2])
        self.ball_grad_scale = (
            np.linalg.norm([self.field.width / 2, self.field.length / 2]) / 4
        )
        self.energy_scale = (
            160 * 4
        ) * 1000  # max wheel speed (rad/s) * 4 wheels * steps

        # Limit robot speeds
        self.max_v = 2.5  # robot max velocity
        self.max_w = 10  # max angular velocity
        self.kick_speed_x = 5.0  # kick speed

        # set starting formation style for
        self.blue_formation = (
            BLUE_START_ONE if not blue_starting_formation else blue_starting_formation
        )
        self.yellow_formation = (
            YELLOW_START_ONE
            if not yellow_starting_formation
            else yellow_starting_formation
        )

        print(f"{n_robots_blue}v{n_robots_yellow} SSL Environment Initialized")

    def teleport_ball(self, x: float, y: float):
        """
        teleport ball to new position in meters

        Note: this does not create a new frame, but mutates the current frame
        """
        ball = Ball(x=x, y=y)
        self.frame.ball = ball
        self.rsim.reset(self.frame)

    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: bool,
        x: float,
        y: float,
        theta: float = None,
    ):
        """
        teleport robot to new position in meters, radians

        Note: this does not create a new frame, but mutates the current frame
        """
        if theta is None:
            if is_team_yellow:
                theta = self.frame.robots_yellow[robot_id].theta
            else:
                theta = self.frame.robots_blue[robot_id].theta
        else:
            theta = rad_to_deg(theta)

        robot = Robot(yellow=is_team_yellow, id=robot_id, x=x, y=y, theta=theta)
        if is_team_yellow:
            self.frame.robots_yellow[robot_id] = robot
        else:
            self.frame.robots_blue[robot_id] = robot

        self.rsim.reset(self.frame)

    def reset(self, *, seed=None, options=None):
        self.reward_shaping_total = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)

        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _frame_to_observations(self) -> Tuple[FrameData, RobotInfo, RobotInfo]:
        """
        return observation data that aligns with grSim

        Returns (vision_observation, yellow_robot_feedback, blue_robot_feedback)
        vision_observation: closely aligned to SSLVision that returns a FramData object
        yellow_robots_info: feedback from individual yellow robots that returns a List[RobotInfo]
        blue_robots_info: feedback from individual blue robots that returns a List[RobotInfo]
        """
        # Ball observation shared by all robots
        ball_obs = BallData(
            self.frame.ball.x * 1e3, self.frame.ball.y * 1e3, self.frame.ball.z * 1e3
        )

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
        return (
            FrameData(self.time_step * self.steps, yellow_obs, blue_obs, [ball_obs]),
            yellow_robots_info,
            blue_robots_info,
        )

    def _get_robot_observation(self, robot):
        robot_pos = RobotData(
            robot.x * 1e3, robot.y * 1e3, float(deg_to_rad(robot.theta))
        )
        robot_info = RobotInfo(robot.infrared)
        return robot_pos, robot_info

    def _get_commands(self, actions) -> list[Robot]:
        commands = []

        for i in range(self.n_robots_blue):
            v_x = actions["team_blue"][i][0]
            v_y = actions["team_blue"][i][1]
            v_theta = actions["team_blue"][i][2]
            cmd = Robot(
                yellow=False,  # Blue team
                id=i,  # ID of the robot
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=self.kick_speed_x if actions["team_blue"][i][3] > 0 else 0.0,
                dribbler=True if actions["team_blue"][i][4] > 0 else False,
            )
            commands.append(cmd)

        for i in range(self.n_robots_yellow):
            v_x = actions["team_yellow"][i][0]
            v_y = actions["team_yellow"][i][1]
            v_theta = actions["team_yellow"][i][2]
            cmd = Robot(
                yellow=True,  # Yellow team
                id=i,  # ID of the robot
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=self.kick_speed_x if actions["team_yellow"][i][3] > 0 else 0.0,
                dribbler=True if actions["team_yellow"][i][4] > 0 else False,
            )
            commands.append(cmd)

        return commands

    def convert_actions(self, action):
        """Clip to absolute max and convert to local"""
        v_x = action[0]
        v_y = action[1]
        v_theta = action[2]
        # clip by max absolute
        # TODO: Not sure if clipping it this way makes sense. We'll see.
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c

        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            # Initialize reward shaping dictionary (info)
            self.reward_shaping_total = {
                "blue_team": {
                    "goal": 0,
                    "rbt_in_gk_area": 0,
                    "done_ball_out": 0,
                    "done_ball_out_right": 0,
                    "done_rbt_out": 0,
                    "energy": 0,
                },
                "yellow_team": {
                    "conceded_goal": 0,
                    "rbt_in_gk_area": 0,
                    "done_ball_out": 0,
                    "done_ball_out_right": 0,
                    "done_rbt_out": 0,
                    "energy": 0,
                },
            }

        # reward_blue = 0
        # reward_yellow = 0
        done = False

        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2

        ball = self.frame.ball

        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid

        # Check if any robot on the blue team exited field or violated rules (for info)
        for (_, robot_b), (_, robot_y) in zip(
            self.frame.robots_blue.items(), self.frame.robots_yellow.items()
        ):
            if abs(robot_y.y) > half_wid or abs(robot_y.x) > half_len:
                done = True
                self.reward_shaping_total["blue_team"]["done_rbt_out"] += 1
            elif abs(robot_y.y) > half_wid or abs(robot_y.x) > half_len:
                done = True
                self.reward_shaping_total["yellow_team"]["done_rbt_out"] += 1
            elif robot_in_gk_area(robot_b):
                done = True
                self.reward_shaping_total["blue_team"]["rbt_in_gk_area"] += 1
            elif robot_in_gk_area(robot_y):
                done = True
                self.reward_shaping_total["yellow_team"]["rbt_in_gk_area"] += 1

        # Check if ball exited field or a goal was made (if blue was attacking)
        # TODO: Add reward shaping for yellow team (obtaining possession of the ball)
        if abs(ball.y) > half_wid or abs(ball.x) > half_len:
            done = True
            self.reward_shaping_total["blue_team"]["done_ball_out"] += 1
        # if the ball is outside the attacking half for blue team (right half of the field)
        elif ball.x > half_len:
            done = True
            # if the ball is inside the goal area otherwise it is a ball out from goalie line
            if abs(ball.y) < half_goal_wid:
                reward_blue = 5
                reward_yellow = -5
                self.reward_shaping_total["blue_team"]["goal"] += 1
                self.reward_shaping_total["yellow_team"]["conceded_goal"] += 1
            else:
                reward = 0
                self.reward_shaping_total["team_blue"]["done_ball_out_right"] += 1
        # elif self.last_frame is not None:

        # Example: Energy penalty for all blue robots
        # total_energy_rw_b = 0
        # total_energy_rw_y = 0
        # for (_, robot_b), (_, robot_y) in zip(
        #     self.frame.robots_blue.items(), self.frame.robots_yellow.items()
        # ):
        #     total_energy_rw_b += self.__energy_pen(robot_b)
        #     total_energy_rw_y += self.__energy_pen(robot_y)

        # avg_energy_rw_b = total_energy_rw_b / len(self.frame.robots_blue)
        # avg_energy_rw_y = total_energy_rw_y / len(self.frame.robots_yellow)

        # energy_rw_b = -(avg_energy_rw_b / self.energy_scale)
        # energy_rw_y = -(avg_energy_rw_y / self.energy_scale)

        # self.reward_shaping_total["blue_team"]["energy"] += energy_rw_b
        # self.reward_shaping_total["yellow_team"]["energy"] += energy_rw_y

        # # Total reward (Scoring reward + Energy penalty v )
        # reward_blue = reward_blue + energy_rw_b
        # reward_yellow = reward_yellow + energy_rw_y

        # reward = {"blue_team": reward_blue, "yellow_team": reward_yellow}

        reward = 0  # NB: We are not using reward for now

        return reward, done

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame (random placement)"""
        pos_frame: Frame = Frame()

        for i in range(self.n_robots_blue):
            x, y, heading = self.blue_formation[i]
            pos_frame.robots_blue[i] = Robot(
                id=i, x=x / 1e3, y=y / 1e3, theta=rad_to_deg(heading)
            )

        for i in range(self.n_robots_yellow):
            x, y, heading = self.yellow_formation[i]
            pos_frame.robots_yellow[i] = Robot(
                id=i, x=x / 1e3, y=y / 1e3, theta=rad_to_deg(heading)
            )

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

    # def __energy_pen(self, robot):
    #     # Sum of abs each wheel speed sent
    #     energy = (
    #         abs(robot.v_wheel0)
    #         + abs(robot.v_wheel1)
    #         + abs(robot.v_wheel2)
    #         + abs(robot.v_wheel3)
    #     )

    #     return energy
