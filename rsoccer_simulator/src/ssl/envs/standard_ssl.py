import math
import random
from typing import Dict

import gymnasium as gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


class SSLStandardEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal on a field with static defenders


    Description:
        The controlled robot is started on the field center and needs to
        score on the positive side field.
    Observation:
        Type: Box(4 + 8*n_robots_blue + 8*n_robots_yellow)
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        +8*i    id i Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta, infra_red]
        +8*i     id i Yellow Robot [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta, infra_red]
    Actions:
        Type: Box(5, )
        Num     Action
        0       id 0 Blue Global X Direction Speed  (%)
        1       id 0 Blue Global Y Direction Speed  (%)
        2       id 0 Blue Angular Speed  (%)
        3       id 0 Blue Kick x Speed  (%)
        4       id 0 Blue Dribbler  (%) (true if % is positive)

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
        field_type=2,
        render_mode="human",
        n_robots_blue=6,
        n_robots_yellow=6,
        time_step=0.025,
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

        print(f"{n_robots_blue}v{n_robots_yellow} SSL Environment Initialized")

    def reset(self, *, seed=None, options=None):
        self.reward_shaping_total = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # Apply the actions to all robots in both teams
        for i in range(self.n_robots_blue):
            self._apply_action(i, action["team_blue"][i], self.frame.robots_blue[i])
        for i in range(self.n_robots_yellow):
            self._apply_action(i, action["team_yellow"][i], self.frame.robots_yellow[i])

        # Proceed with step calculations (including reward and done check)
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _apply_action(self, robot_id, action, robot):
        # Convert and apply movement and control actions for each robot
        angle = robot.theta
        v_x, v_y, v_theta = self.convert_actions(action, np.deg2rad(angle))

        robot.v_x = v_x
        robot.v_y = v_y
        robot.v_theta = v_theta
        robot.kick_v_x = self.kick_speed_x if action[3] > 0 else 0.0
        robot.dribbler = True if action[4] == 0 else False

    def _frame_to_observations(self):
        # Ball observation shared by all robots
        ball_obs = [
            self.norm_pos(self.frame.ball.x),
            self.norm_pos(self.frame.ball.y),
            self.norm_v(self.frame.ball.v_x),
            self.norm_v(self.frame.ball.v_y),
        ]

        # Robots observation (Blue + Yellow)
        robots_obs = []
        for robot in self.frame.robots_blue.values():
            robots_obs.extend(self._get_robot_observation(robot))
        for robot in self.frame.robots_yellow.values():
            robots_obs.extend(self._get_robot_observation(robot))

        # Return the complete shared observation
        return np.array(ball_obs + robots_obs, dtype=np.float32)

    def _get_robot_observation(self, robot):
        return [
            self.norm_pos(robot.x),
            self.norm_pos(robot.y),
            np.sin(np.deg2rad(robot.theta)),
            np.cos(np.deg2rad(robot.theta)),
            self.norm_v(robot.v_x),
            self.norm_v(robot.v_y),
            self.norm_w(robot.v_theta),
            1 if robot.infrared else 0,
        ]

    def _get_commands(self, actions):
        commands = []

        for i in range(self.n_robots_blue):
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(
                actions["team_blue"][i], np.deg2rad(angle)
            )
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
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(
                actions["team_yellow"][i], np.deg2rad(angle)
            )
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

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), -v_x * np.sin(
            angle
        ) + v_y * np.cos(angle)

        # clip by max absolute
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

        reward_blue = 0
        reward_yellow = 0
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
            if robot_b.x < -0.2 or abs(robot_b.y) > half_wid:
                done = True
                self.reward_shaping_total["blue_team"]["done_rbt_out"] += 1
            elif robot_y.x > 0.2 or abs(robot_y.y) > half_wid:
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
        if ball.x < 0 or abs(ball.y) > half_wid:
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
        elif self.last_frame is not None:
            # # TODO: Creating non stopping reward functions (rewards that are calculated per action) ->

            # Example: Energy penalty for all blue robots
            total_energy_rw_b = 0
            total_energy_rw_y = 0
            for (_, robot_b), (_, robot_y) in zip(
                self.frame.robots_blue.items(), self.frame.robots_yellow.items()
            ):
                total_energy_rw_b += self.__energy_pen(robot_b)
                total_energy_rw_y += self.__energy_pen(robot_y)

            avg_energy_rw_b = total_energy_rw_b / len(self.frame.robots_blue)
            avg_energy_rw_y = total_energy_rw_y / len(self.frame.robots_yellow)

            energy_rw_b = -(avg_energy_rw_b / self.energy_scale)
            energy_rw_y = -(avg_energy_rw_y / self.energy_scale)

            self.reward_shaping_total["blue_team"]["energy"] += energy_rw_b
            self.reward_shaping_total["yellow_team"]["energy"] += energy_rw_y

            # Total reward (Scoring reward + Energy penalty v )
            reward_blue = reward_blue + energy_rw_b
            reward_yellow = reward_yellow + energy_rw_y

        reward = {"blue_team": reward_blue, "yellow_team": reward_yellow}

        return reward, done

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame (random placement)"""
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

    def __energy_pen(self, robot):
        # Sum of abs each wheel speed sent
        energy = (
            abs(robot.v_wheel0)
            + abs(robot.v_wheel1)
            + abs(robot.v_wheel2)
            + abs(robot.v_wheel3)
        )

        return energy
