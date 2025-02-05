import numpy as np

from motion_planning.src.pid.pid import TwoDPID
from global_utils.math_utils import distance, normalise_heading
from robot_control.src.utils.shooting_utils import find_best_shot
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import Game, Field
from entities.data.command import RobotCommand
from entities.data.vision import RobotData, BallData
from robot_control.src.skills import (
    align_defenders,
    face_ball,
    find_likely_enemy_shooter,
    get_goal_centre,
    kick_ball,
    go_to_ball,
    to_defense_parametric,
    turn_on_spot,
    empty_command,
    go_to_point,
    velocity_to_orientation,
)
from motion_planning.src.pid import PID
from typing import List, Tuple
from math import dist
import logging

from team_controller.src.controllers.common.robot_controller_abstract import AbstractRobotController
from team_controller.src.controllers.sim.rsim_robot_controller import RSimRobotController

logger = logging.getLogger(__name__)


from robot_control.src.utils.passing_utils import calculate_adjusted_receiver_pos


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
        self.dist_tolerance = 0.05
        self.ball_in_flight = False
        self.ball_traj_points = []
        self.ball_launch_pos = None

    def enact(self, passer_has_ball: bool) -> Tuple[RobotCommand, RobotCommand]:
        """
        return the command for passer and receiver in that order.
        """

        # TODO: need to ensure this func works when ball_data or robot_pos is None
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

            # TODO: add line filtering to calculate the adjusted position
            if ball_data is not None:
                self.ball_traj_points.append((ball_data.x, ball_data.y))
            adjusted_pos = calculate_adjusted_receiver_pos(
                receiver_data, self.ball_traj_points
            )  # we are assuming the adjusted position should be extremely close
            catch_orientation = np.arctan2(
                ball_data.y - adjusted_pos[1], ball_data.x - adjusted_pos[0]
            )
            # catch_orientation = np.arctan2(
            #     ball_data.y - receiver_data.y, ball_data.x - receiver_data.x
            # )
            receiver_cmd = go_to_point(
                self.pid_oren,
                self.pid_trans,
                receiver_data,
                self.receiver_id,
                adjusted_pos,
                catch_orientation,
                dribbling=True
            )

        else:
            catch_orientation = normalise_heading(shot_orientation + np.pi)
            if (
                distance((receiver_data.x, receiver_data.y), self.target_coords)
                < self.dist_tolerance
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
            if ball_data is not None:
                self.ball_traj_points.append((ball_data.x, ball_data.y))
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
        best_shot, size_of_shot = find_best_shot(
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

def defend(
    pid_oren: PID,
    pid_2d: TwoDPID,
    game: Game,
    is_yellow: bool,
    defender_id: int,
    env,
) -> RobotCommand:
    # Assume that is_yellow <-> not is_left here # TODO : FIX
    friendly, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
    shooters_data = find_likely_enemy_shooter(enemy, balls)
    orientation = None
    tracking_ball = False


    # if game.in_box(balls[0].x, balls[0].y):
    #     print("AAAAAAAAAAAAAAAAAAAAAA")
    #     # return None

    if not shooters_data:
        target_tracking_coord = balls[0].x, balls[0].y
        # TODO game.get_ball_velocity() can return (None, None)
        if (
            game.get_ball_velocity() is not None
            and None not in game.get_ball_velocity()
        ):
            orientation = velocity_to_orientation(game.get_ball_velocity())
            tracking_ball = True
    else:
        # TODO (deploy more defenders, or find closest shooter?)
        sd = shooters_data[0]
        target_tracking_coord = sd.x, sd.y
        orientation = sd.orientation

    real_def_pos = friendly[defender_id].x, friendly[defender_id].y
    current_def_parametric = to_defense_parametric(real_def_pos, is_left=not is_yellow)
    target = align_defenders(
        current_def_parametric, target_tracking_coord, orientation, not is_yellow, env
    )
    cmd = go_to_point(
        pid_oren,
        pid_2d,
        friendly[defender_id],
        defender_id,
        target,
        face_ball(real_def_pos, (balls[0].x, balls[0].y)),
        dribbling=True,
    )

    gp = get_goal_centre(is_left=not is_yellow)
    env.draw_line(
        [gp, (target_tracking_coord[0], target_tracking_coord[1])],
        width=5,
        color="RED" if tracking_ball else "PINK",
    )

    return cmd
