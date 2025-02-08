import numpy as np

from typing import Optional
from robot_control.src.utils.shooting_utils import find_best_shot
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import Game, Field
from entities.game.game_object import Colour
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
        
        if best_shot is None:
            # print("No shot found")
            return None
        
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

def is_goal_blocked(game: Game) -> bool:
    """
    Determines whether the goal is blocked by enemy robots.

    :param game: The game state containing robot and ball positions.
    :return: True if the goal is blocked, False otherwise.
    """
    ball_x, ball_y = game.ball.x, game.ball.y
    goal_x = -4.5
    goal_y_range = (1, -1)  # Goalposts' y-range

    # Define the line equation from ball to goal
    def is_point_on_line(point, start, end, tolerance=0.1):
        """Check if a point is approximately on the line segment from start to end."""
        start = np.array(start)
        end = np.array(end)
        point = np.array(point)
        line_vec = end - start
        point_vec = point - start
        proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        projected_point = start + proj * line_vec
        
        return np.linalg.norm(point - projected_point) < tolerance and 0 <= proj <= 1

    # Check if any enemy robot is between the ball and goal
    for enemy in game.enemy_robots:
        enemy_x, enemy_y = enemy.x, enemy.y  # Extract coordinates properly
        
        # Check if the enemy is between the ball and goal in the x-range
        if ball_x > enemy_x > goal_x:
            # Check if the enemy is within the goal's y-range
            if goal_y_range[1] <= enemy_y <= goal_y_range[0]:  
                # Check if the enemy is on the ball-to-goal path
                if is_point_on_line((enemy_x, enemy_y), (ball_x, ball_y), (goal_x, np.mean(goal_y_range))):
                    return True

    return False

# intent on scoring goal
def score_goal_one_v_one(
    game_obj: Game,
    shooter_id: int,
    pid_oren: PID,
    pid_trans: PID,
    shoot_at_goal_colour: Optional[Colour] = None,
) -> RobotCommand:
    """
    shoot_at_goal_colour should only be used i
    """
    if not shoot_at_goal_colour:
        shoot_at_goal_colour = (
            Colour.BLUE if game_obj.my_team_is_yellow else Colour.YELLOW
        )

    target_goal_line = game_obj.field.enemy_goal_line(
        shoot_at_goal_colour == Colour.BLUE
    )
    
    print(shoot_at_goal_colour)

    # If no frame data, skip
    if not game_obj.get_latest_frame():
        return

    friendly_robots = game_obj.friendly_robots
    enemy_robots = game_obj.enemy_robots
    ball = game_obj.ball
    # According to how game works, we take the most confident ball

    goal_x = target_goal_line.coords[0][0]
    goal_y1 = target_goal_line.coords[1][1]
    goal_y2 = target_goal_line.coords[0][1]

    # calculate best shot from the position of the ball
    # TODO: add sampling function to try to find other angles to shoot from that are more optimal
    if friendly_robots and enemy_robots and ball:
        best_shot = find_best_shot(
            ball, enemy_robots, goal_x, goal_y1, goal_y2, shoot_at_goal_colour
        )

        if best_shot is None:
            return None
        
        shot_orientation = np.atan2((best_shot - ball.y), (goal_x - ball.x))

        robot_data: RobotData = (
            friendly_robots[shooter_id].robot_data
            if shooter_id < len(friendly_robots)
            else None
        )

        # ball_data: BallData = ball.ball_data

        if ball is not None and robot_data is not None:
            if robot_data is not None:
                logging.debug("robot has ball")
                current_oren = robot_data.orientation

                # if robot has ball and is facing the goal, kick the ball
                # TODO: This should be changed to a smarter metric (ie within the range of tolerance of the shot)
                # Because 0.02 as a threshold is meaningless (different at different distances)
                # TODO: consider also adding a distance from goal threshold
                # print(current_oren, shot_orientation)
                if abs(current_oren - shot_orientation) % np.pi <= 0.05 and not is_goal_blocked(game_obj):
                    logger.info("kicking ball")
                    robot_command = kick_ball()
                # else, robot has ball, but needs to turn to the right direction
                # TODO: Consider also advancing closer to the goal
                elif is_goal_blocked(game_obj):
                    return None
                else:
                    robot_command = turn_on_spot(
                        pid_oren,
                        pid_trans,
                        robot_data,
                        shooter_id,
                        shot_orientation,
                        dribbling=True,
                        pivot_on_ball=True,
                    )

    return robot_command