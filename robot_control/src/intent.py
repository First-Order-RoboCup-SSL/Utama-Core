import numpy as np

from robot_control.src.utils.shooting_utils import find_best_shot
from entities.game import Game, Field
from entities.data.command import RobotCommand
from entities.data.vision import RobotData, BallData
from robot_control.src.skills import kick_ball, go_to_ball, turn_on_spot
from motion_planning.src.pid import PID


# intent on scoring goal
def score_goal(
    game_obj: Game,
    shooter_has_ball: bool,
    shooter_id: int,
    pid_oren: PID,
    pid_trans: PID,
) -> RobotCommand:
    target_goal_line = game_obj.field.enemy_goal_line
    latest_frame = game_obj.get_my_latest_frame()
    if latest_frame:
        friendly_robots, enemy_robots, balls = latest_frame

    # TODO: Not sure if this is sufficient for both blue and yellow scoring
    goal_x = target_goal_line.coords[0][0]
    goal_y1 = target_goal_line.coords[1][1]
    goal_y2 = target_goal_line.coords[0][1]

    # calculate best shot from the position of the ball
    # TODO: add sampling function to try to find other angles to shoot from that are more optimal
    if friendly_robots and enemy_robots and balls:
        best_shot = find_best_shot(balls[0], enemy_robots, goal_x, goal_y1, goal_y2)

        shot_orientation = np.atan2((best_shot - balls[0].y), (goal_x - balls[0].x))

        robot_data: RobotData = (
            friendly_robots[shooter_id] if shooter_id < len(friendly_robots) else None
        )

        # TODO: For now we just look at the first ball, but this will eventually have to be smarter
        ball_data: BallData = balls[0]

        if ball_data is not None and robot_data is not None:
            if robot_data is not None:
                if shooter_has_ball:
                    print("robot has ball")
                    current_oren = robot_data.orientation

                    # if robot has ball and is facing the goal, kick the ball
                    # TODO: This should be changed to a smarter metric (ie within the range of tolerance of the shot)
                    # Because 0.02 as a threshold is meaningless (different at different distances)
                    # TODO: consider also adding a distance from goal threshold
                    if (
                        abs(np.round(current_oren, 2) - np.round(shot_orientation, 2))
                        <= 0.02
                    ):
                        print("kicking ball")
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
                    print("approaching ball")
                    robot_command = go_to_ball(
                        pid_oren, pid_trans, robot_data, shooter_id, ball_data
                    )

    return robot_command
