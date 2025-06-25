from entities.game import Game
from motion_planning.src.pid import PID, TwoDPID
from typing import Tuple
from entities.data.command import RobotCommand
from skills.src.go_to_point import go_to_point


def goalkeep(
    game: Game,
    pid_oren: PID,
    pid_trans: TwoDPID,
    robot_id: int,
):
    robot_data = game.get_robot_pos(is_yellow, robot_id)
    if goalie_has_ball:
        target_oren = 0 if is_left_goal else math.pi
        print("TARGET OREN", target_oren)
        return go_to_point(
            pid_oren,
            pid_trans,
            robot_data,
            robot_id,
            ((-4 if is_left_goal else 4), 0),
            target_oren,
            True,
        )

    if is_left_goal:
        target = game.predict_ball_pos_at_x(-4.5)
    else:
        target = game.predict_ball_pos_at_x(4.5)

    if not target or abs(target[1]) > 0.5:
        target = (-4.5 if is_left_goal else 4.5, 0)

    target and not find_likely_enemy_shooter(
        game.get_robots_pos(not is_yellow), [game.ball]
    ):
        cmd = go_to_point(
            game,
            pid_oren,
            pid_trans,
            robot_id,
            target,
            dribbling=True,
        )
    return cmd
