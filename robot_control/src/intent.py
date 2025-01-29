import numpy as np

from global_utils.math_utils import squared_distance, normalise_heading
from robot_control.src.utils.shooting_utils import find_best_shot, find_shot_quality
from robot_control.src.utils.pass_quality_utils import (
    find_pass_quality,
    find_best_receiver_position,
)
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import Game, Field
from entities.data.command import RobotCommand
from entities.data.vision import RobotData, BallData
from robot_control.src.skills import (
    kick_ball,
    go_to_ball,
    turn_on_spot,
    empty_command,
    go_to_point,
)
from motion_planning.src.pid import PID
from typing import List, Tuple, Dict, Optional
from math import dist
import logging

logger = logging.getLogger(__name__)


from robot_control.src.utils.passing_utils import calculate_adjusted_receiver_pos


class Play2v5:
    def __init__(
        self,
        pid_oren: PID,
        pid_trans: PID,
        game: Game,
        robot_ids: List[int],
        shoot_in_left_goal: bool,
        pass_quality_thresh: float,
        shot_quality_thresh: float,
    ):
        self.pid_oren = pid_oren
        self.pid_trans = pid_trans
        self.game = game
        self.robot_ids = robot_ids
        self.shoot_in_left_goal = shoot_in_left_goal
        self.pass_quality_thresh = pass_quality_thresh
        self.shot_quality_thresh = shot_quality_thresh
        self.my_team_is_yellow = game.my_team_is_yellow

    def intercept_ball(
        self,
        receiver_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        ball_vel: Tuple[float, float],
        robot_speed: float,
    ) -> Tuple[float, float]:
        """
        Simple function to calculate intercept position for a robot.
        Assumes the ball is moving in a straight line, and we find the point where the receiver should go.
        """
        # Calculate the time it will take for the robot to reach the ball (simplified)
        distance_to_ball = np.linalg.norm(np.array(ball_pos) - np.array(receiver_pos))
        time_to_reach = distance_to_ball / robot_speed  # Assuming constant robot speed

        # Predict the future position of the ball
        intercept_pos = (
            ball_pos[0] + ball_vel[0] * time_to_reach,
            ball_pos[1] + ball_vel[1] * time_to_reach,
        )

        return intercept_pos

    def enact(self, ball_possessor_id: Optional[int]) -> Dict[int, RobotCommand]:
        """
        Returns a dict {robot_id: command} for each robot.
        """
        commands = {}

        target_goal_line = self.game.field.enemy_goal_line(self.my_team_is_yellow)
        latest_frame = self.game.get_my_latest_frame(self.my_team_is_yellow)
        if not latest_frame:
            return
        friendly_robots, enemy_robots, balls = latest_frame

        # TODO: Not sure if this is sufficient for both blue and yellow scoring
        # It won't be because note that in real life the blue team is not necessarily
        # on the left of the pitch
        goal_x = target_goal_line.coords[0][0]
        goal_y1 = target_goal_line.coords[1][1]
        goal_y2 = target_goal_line.coords[0][1]

        # TODO: For now we just look at the first ball, but this will eventually have to be smarter
        ball_data: BallData = balls[0]

        ### CASE 1: No one has the ball → Try to intercept it ###
        if ball_possessor_id is None:
            best_interceptor = None
            best_intercept_score = float("inf")  # Lower is better (closer to ball path)

            for robot in friendly_robots:

                ball_pos = (ball_data.x, ball_data.y)
                ball_vel = (ball_data.vx, ball_data.vy)

                # this is a bit hacky for now. we need a better interception function
                # Calculate intercept position using the intercept_ball function
                intercept_pos = self.intercept_ball(
                    robot, ball_pos, ball_vel, robot_speed=4.0
                )  # Use appropriate robot speed

                # Calculate how close the robot is to the intercept position (lower score is better)
                intercept_score = squared_distance(robot, intercept_pos)

                if intercept_score < best_intercept_score:
                    best_interceptor = rid
                    best_intercept_score = intercept_score

            # Send the best robot to intercept
            if best_interceptor is not None:
                intercept_pos = self.intercept_ball(
                    (
                        friendly_robots[best_interceptor].x,
                        friendly_robots[best_interceptor].y,
                    ),
                    (ball_data.x, ball_data.y),
                    (ball_data.vx, ball_data.vy),
                    robot_speed=4.0,
                )
                commands[best_interceptor] = go_to_point(
                    self.pid_oren,
                    self.pid_trans,
                    friendly_robots[best_interceptor],
                    best_interceptor,
                    intercept_pos,
                )

            return commands  # Only interceptors act, others wait

        ### CASE 2: Someone has the ball ###
        possessor_data = friendly_robots[ball_possessor_id]

        # Check shot opportunity
        shot_quality = find_shot_quality(
            possessor_data,
            enemy_robots,
            goal_x,
            goal_y1,
            goal_y2,
            self.shoot_in_left_goal,
        )
        if shot_quality > self.shot_quality_thresh:
            commands[ball_possessor_id] = kick_ball()
            return commands  # Just shoot, no need to pass

        # Check for best pass
        best_receiver_id = None
        best_pass_quality = 0
        for rid in self.robot_ids:
            if rid == ball_possessor_id:
                continue
            pq = find_pass_quality(ball_possessor_id, rid)
            if pq > best_pass_quality:
                best_pass_quality = pq
                best_receiver_id = rid

        if (
            best_receiver_id is not None
            and best_pass_quality > self.pass_quality_thresh
        ):
            commands[ball_possessor_id] = kick_ball()
        else:
            commands[ball_possessor_id] = empty_command(
                dribbler_on=True
            )  # Wait for a better pass

        # Move non-possessing robots to good positions
        for rid in self.robot_ids:
            if rid == ball_possessor_id:
                continue
            target_pos = find_best_receiver_position(rid)
            commands[rid] = go_to_point(
                self.pid_oren,
                self.pid_trans,
                friendly_robots[rid],
                rid,
                target_pos,
            )

        return commands


class PassBall:
    def __init__(
        self,
        pid_oren: PID,
        pid_trans: PID,
        game: Game,
        passer_id: int,
        receiver_id: int,
        target_coords: Tuple[float, float],
    ):
        self.pid_oren = pid_oren
        self.pid_trans = pid_trans
        self.game = game
        self.passer_id = passer_id
        self.receiver_id = receiver_id
        self.target_coords = target_coords
        self.my_team_is_yellow = game.my_team_is_yellow

        self.angle_tolerance = 0.01
        self.sq_dist_tolerance = 0.01
        self.ball_in_flight = False
        self.ball_launch_pos = None

    def enact(self, passer_has_ball: bool) -> Tuple[RobotCommand, RobotCommand]:
        """
        return the command for passer and receiver in that order.
        """
        passer_ready = False
        receiver_ready = False

        passer_data = self.game.get_robot_pos(self.my_team_is_yellow, self.passer_id)
        receiver_data = self.game.get_robot_pos(
            self.my_team_is_yellow, self.receiver_id
        )
        ball_data = self.game.ball

        ### passer commands ###

        passer_oren = passer_data.orientation
        shot_orientation = np.arctan2(
            self.target_coords[1] - ball_data.y,
            self.target_coords[0] - ball_data.x,
        )

        if not passer_has_ball and not self.ball_in_flight:
            passer_cmd = go_to_ball(
                self.pid_oren,
                self.pid_trans,
                passer_data,
                self.passer_id,
                ball_data,
            )

        else:
            if (
                abs(passer_oren - shot_orientation) <= self.angle_tolerance
                or self.ball_in_flight
            ):
                passer_ready = True
                passer_cmd = empty_command(
                    dribbler_on=True
                )  # default action is to wait. Unless both are ready then intiate pass
            else:
                passer_cmd = turn_on_spot(
                    self.pid_oren,
                    self.pid_trans,
                    passer_data,
                    self.passer_id,
                    shot_orientation,
                    dribbling=True,
                    pivot_on_ball=True,
                )

        ### receiver commands ###
        receiver_oren = receiver_data.orientation

        # if ball has already been kicked and heading towards receiver
        if self.ball_in_flight:

            # TODO: fix fine adjustment of receiver position

            # adjusted_pos = calculate_adjusted_receiver_pos(
            #     self.ball_launch_pos, receiver_data, ball_data
            # )  # we are assuming the adjusted position should be extremely close
            # catch_orientation = np.arctan2(
            #     ball_data.y - adjusted_pos[1], ball_data.x - adjusted_pos[0]
            # )
            catch_orientation = np.arctan2(
                ball_data.y - receiver_data.y, ball_data.x - receiver_data.x
            )
            receiver_cmd = go_to_point(
                self.pid_oren,
                self.pid_trans,
                receiver_data,
                self.receiver_id,
                self.target_coords,
                catch_orientation,
            )

        else:
            catch_orientation = normalise_heading(shot_orientation + np.pi)
            if (
                squared_distance((receiver_data.x, receiver_data.y), self.target_coords)
                < self.sq_dist_tolerance
                and abs(
                    receiver_oren - catch_orientation,
                )
                < self.angle_tolerance
            ):

                receiver_cmd = empty_command(dribbler_on=True)
                receiver_ready = True

            else:
                receiver_cmd = go_to_point(
                    self.pid_oren,
                    self.pid_trans,
                    receiver_data,
                    self.receiver_id,
                    self.target_coords,
                    catch_orientation,
                )

        if passer_ready and receiver_ready:
            passer_cmd = kick_ball()
            self.ball_launch_pos = (ball_data.x, ball_data.y)
            self.ball_in_flight = True

        return passer_cmd, receiver_cmd


# intent on scoring goal
def score_goal(
    game_obj: Game,
    shooter_has_ball: bool,
    shooter_id: int,
    pid_oren: PID,
    pid_trans: PID,
    is_yellow: bool,
    shoot_in_left_goal: bool,
) -> RobotCommand:

    target_goal_line = game_obj.field.enemy_goal_line(is_yellow)
    latest_frame = game_obj.get_my_latest_frame(is_yellow)
    if not latest_frame:
        return
    friendly_robots, enemy_robots, balls = latest_frame

    # TODO: Not sure if this is sufficient for both blue and yellow scoring
    # It won't be because note that in real life the blue team is not necessarily
    # on the left of the pitch
    goal_x = target_goal_line.coords[0][0]
    goal_y1 = target_goal_line.coords[1][1]
    goal_y2 = target_goal_line.coords[0][1]

    # calculate best shot from the position of the ball
    # TODO: add sampling function to try to find other angles to shoot from that are more optimal
    if friendly_robots and enemy_robots and balls:
        best_shot = find_best_shot(
            balls[0], enemy_robots, goal_x, goal_y1, goal_y2, shoot_in_left_goal
        )

        shot_orientation = np.atan2((best_shot - balls[0].y), (goal_x - balls[0].x))

        robot_data: RobotData = (
            friendly_robots[shooter_id] if shooter_id < len(friendly_robots) else None
        )

        # TODO: For now we just look at the first ball, but this will eventually have to be smarter
        ball_data: BallData = balls[0]

        if ball_data is not None and robot_data is not None:
            if robot_data is not None:
                if shooter_has_ball:
                    logging.debug("robot has ball")
                    current_oren = robot_data.orientation

                    # if robot has ball and is facing the goal, kick the ball
                    # TODO: This should be changed to a smarter metric (ie within the range of tolerance of the shot)
                    # Because 0.02 as a threshold is meaningless (different at different distances)
                    # TODO: consider also adding a distance from goal threshold
                    if abs(current_oren - shot_orientation) <= 0.01:
                        logger.info("kicking ball")
                        robot_command = kick_ball()
                    # else, robot has ball, but needs to turn to the right direction
                    # TODO: Consider also advancing closer to the goal
                    else:
                        robot_command = turn_on_spot(
                            pid_oren,
                            pid_trans,
                            robot_data,
                            shooter_id,
                            shot_orientation,
                            dribbling=shooter_has_ball,
                            pivot_on_ball=True,
                        )

                else:
                    logger.debug("approaching ball %lf", robot_data.orientation)
                    robot_command = go_to_ball(
                        pid_oren, pid_trans, robot_data, shooter_id, ball_data
                    )

    return robot_command


def find_likely_enemy_shooter(enemy_robots, balls) -> List[RobotData]:
    ans = []
    for ball in balls:
        for er in enemy_robots:
            if dist((er.x, er.y), (ball.x, ball.y)) < 0.2:
                # Ball is close to this robot
                ans.append(er)
    return list(set(ans))
