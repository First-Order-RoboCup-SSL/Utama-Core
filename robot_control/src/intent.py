import numpy as np

from robot_control.src.utils.shooting_utils import find_best_shot
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import Game, Field
from entities.data.command import RobotCommand
from entities.data.vision import RobotData, BallData
from robot_control.src.skills import kick_ball, go_to_ball, turn_on_spot
from motion_planning.src.pid import PID
from typing import List
from math import dist
import logging

logger = logging.getLogger(__name__)

# intent on scoring goal
def score_goal(
    game_obj: Game,
    shooter_has_ball: bool,
    shooter_id: int,
    pid_oren: PID,
    pid_trans: PID,
    is_yellow: bool,
    shoot_in_left_goal: bool 
) -> RobotCommand:

    target_goal_line = game_obj.field.enemy_goal_line(is_yellow)
    latest_frame = game_obj.get_my_latest_frame(is_yellow)
    if latest_frame:
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
        best_shot = find_best_shot(balls[0], enemy_robots, goal_x, goal_y1, goal_y2, shoot_in_left_goal)

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
                    if (
                        abs(current_oren - shot_orientation)
                        <= 0.005
                    ):
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