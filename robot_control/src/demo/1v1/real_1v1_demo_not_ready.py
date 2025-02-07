from typing import Tuple
from motion_planning.src.pid.pid import get_real_pids
from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import score_goal, find_likely_enemy_shooter
from robot_control.src.skills import (
    go_to_point,
    face_ball,
    velocity_to_orientation,
    clamp_to_goal_height,
    predict_goal_y_location,
)
from team_controller.src.controllers import RealRobotController
from entities.game import Game
from entities.data.command import RobotCommand
import time
import queue
import logging
import threading
import math
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data.vision_receiver import VisionDataReceiver
import random

logger = logging.getLogger(__name__)


def data_update_listener(receiver: VisionDataReceiver):
    receiver.pull_game_data()


def ball_out_of_bounds(ball_x: float, ball_y: float) -> bool:
    return abs(ball_x) > 4.5 or abs(ball_y) > 3


def find_goal_position(game: Game) -> Tuple[float, float]:
    if game.my_team_is_yellow:
        goal_center = (4.5, 0)
    else:
        goal_center = (-4.5, 0)

    _, enemy, balls = game.get_my_latest_frame()
    shooters_data = find_likely_enemy_shooter(enemy, balls)

    orientation = None
    if not shooters_data:
        target_tracking_coord = (game.ball.x, game.ball.y)
        if game.get_ball_velocity() and None not in game.get_ball_velocity():
            orientation = velocity_to_orientation(game.get_ball_velocity())
    else:
        sd = shooters_data[0]
        target_tracking_coord = (sd.x, sd.y)
        orientation = sd.orientation

    if orientation is None:
        return goal_center
    else:
        return (
            goal_center[0],
            clamp_to_goal_height(
                predict_goal_y_location(
                    target_tracking_coord, orientation, game.my_team_is_yellow
                )
            ),
        )


def improved_block_goal_and_attacker(
    robot_pose, attacker_pose, ball_pose, game, pid_oren, pid_trans, attacker_has_ball
):
    rx, ry, rtheta = robot_pose
    ax, ay = attacker_pose[:2]
    bx, by = ball_pose[:2]

    if attacker_has_ball:
        gx, gy = find_goal_position(game)
        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            target_x, target_y = gx, gy
        else:
            target_x = ax + 0.4 * agx
            target_y = ay + 0.4 * agy
        face_theta = math.atan2(ay - ry, ax - rx)
    else:
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        dist_to_ball = math.hypot(rx - bx, ry - by)
        if dist_to_ball > 1.0:
            ratio = 1.0 / dist_to_ball
            target_x = rx + (bx - rx) * ratio
            target_y = ry + (by - ry) * ratio
        face_theta = math.atan2(by - ry, bx - rx)

    cmd = go_to_point(
        pid_oren, pid_trans, (rx, ry, rtheta), 0, (target_x, target_y), face_theta
    )
    return cmd


def real_world_strat_atk(
    game: Game, controller: RealRobotController, stop_event: threading.Event
):
    pid_oren, pid_trans = get_real_pids(6)

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True
    data_thread.start()

    dribbling = False
    dribble_task = None
    ROBOT_ID = 0

    while not stop_event.is_set():
        if not message_queue.empty():
            msg_type, msg = message_queue.get()
            if msg_type == MessageType.VISION:
                game.add_new_state(msg)

        robot = game.friendly_robots[ROBOT_ID]
        ball = game.ball
        enemy = game.enemy_robots[0] if len(game.enemy_robots) > 0 else None

        if not enemy:
            continue

        has_ball = (
            math.hypot(robot.x - ball.x, robot.y - ball.y) < 0.1
            and abs(robot.orientation - math.atan2(ball.y - robot.y, ball.x - robot.x))
            < 0.2
        )

        if has_ball:
            cmd = score_goal(
                game, True, ROBOT_ID, pid_oren, pid_trans, game.my_team_is_yellow, False
            )
            if cmd is None:
                target = dribble_to_target_decision_maker(game, ROBOT_ID, controller)
                if target:
                    if not dribble_task:
                        dribble_task = DribbleToTarget(
                            pid_oren, pid_trans, game, ROBOT_ID, target
                        )
                    cmd = dribble_task.enact(has_ball)
                    dribbling = True
        else:
            cmd = go_to_point(
                pid_oren,
                pid_trans,
                (robot.x, robot.y, robot.orientation),
                ROBOT_ID,
                (ball.x, ball.y),
                face_ball((robot.x, robot.y), (ball.x, ball.y)),
                dribbling=True,
            )

        if cmd:
            controller.add_robot_commands(cmd, ROBOT_ID)
            controller.send_robot_commands()


def real_world_strat_def(
    game: Game, controller: RealRobotController, stop_event: threading.Event
):
    pid_oren, pid_trans = get_real_pids(6)

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True
    data_thread.start()

    ROBOT_ID = 0

    while not stop_event.is_set():
        if not message_queue.empty():
            msg_type, msg = message_queue.get()
            if msg_type == MessageType.VISION:
                game.add_new_state(msg)

        robot = game.friendly_robots[ROBOT_ID]
        enemy = game.enemy_robots[0] if len(game.enemy_robots) > 0 else None
        ball = game.ball

        if not enemy:
            continue

        # Safety check
        if math.hypot(robot.x - enemy.x, robot.y - enemy.y) < EMERGENCY_STOP_DISTANCE:
            controller.add_robot_commands(RobotCommand(0, 0, 0), ROBOT_ID)
            controller.send_robot_commands()
            continue

        enemy_has_ball = math.hypot(enemy.x - ball.x, enemy.y - ball.y) < 0.1
        cmd = improved_block_goal_and_attacker(
            (robot.x, robot.y, robot.orientation),
            (enemy.x, enemy.y),
            (ball.x, ball.y),
            game,
            pid_oren,
            pid_trans,
            enemy_has_ball,
        )

        if cmd:
            controller.add_robot_commands(cmd, ROBOT_ID)
            controller.send_robot_commands()


def dribble_to_target_decision_maker(
    game: Game,
    robot_id: int,
    grsim_controller: RealRobotController,
    safe_distance: float = 0.1,
) -> Tuple[float, float]:
    """
    Determines the optimal (x, y) position for the robot to dribble to,
    avoiding interception by the enemy.

    :param game: The game object containing current state information.
    :param robot_id: The ID of the robot making the decision.
    :param goal_pos: The (x, y) position of the goal to score in.
    :param enemy_robot_pos: The (x, y) position of the enemy robot.
    :param ball_pos: The (x, y) position of the ball.
    :param safe_distance: The minimum distance to maintain from the enemy robot.
    :return: The optimal (x, y) position for the robot to dribble to.
    """
    robot = game.friendly_robots[robot_id]  # Assuming the robot is on the yellow team

    if game._my_team_is_yellow:
        goal_x, goal_y = -4.5, 0
    else:
        goal_x, goal_y = 4.5, 0

    # If the robot has the ball, move towards the goal while avoiding the enemy
    if grsim_controller.robot_has_ball(robot_id):
        # Calculate the direction vector from the robot to the goal
        goal_dx = goal_x - robot.x
        goal_dy = goal_y - robot.y
        goal_dist = math.hypot(goal_dx, goal_dy)

        # Calculate the direction vector from the enemy to the robot
        smallest_enemy_dist = 0
        enemy_dx = 0
        enemy_dy = 0
        for enemy_robot in game.enemy_robots:
            if enemy_robot.robot_data is not None:
                enemy_dx = robot.x - enemy_robot.x
                enemy_dy = robot.y - enemy_robot.y
                enemy_dist = math.hypot(enemy_dx, enemy_dy)
                if smallest_enemy_dist == 0 or enemy_dist < smallest_enemy_dist:
                    smallest_enemy_dist = enemy_dist
                    enemy_dx = enemy_dx
                    enemy_dy = enemy_dy

        # If the enemy is too close, adjust the target position to avoid interception
        if enemy_dist > safe_distance:
            # Move perpendicular to the enemy-robot line to avoid the enemy
            perpendicular_dx = -enemy_dy
            perpendicular_dy = enemy_dx
            perpendicular_dist = math.hypot(perpendicular_dx, perpendicular_dy)

            # Normalize the perpendicular vector
            if perpendicular_dist > 0:
                perpendicular_dx /= perpendicular_dist
                perpendicular_dy /= perpendicular_dist

            # Move away from the enemy while still progressing towards the goal
            target_x = robot.x + perpendicular_dx * 100 + goal_dx * 0.4
            target_y = robot.y + perpendicular_dy * 100 + goal_dy * 0.4
        else:
            # Move directly towards the goal
            # Move perpendicular to the enemy-robot line to avoid the enemy
            perpendicular_dx = -enemy_dy
            perpendicular_dy = enemy_dx
            perpendicular_dist = math.hypot(perpendicular_dx, perpendicular_dy)

            # Normalize the perpendicular vector
            if perpendicular_dist > 0:
                perpendicular_dx /= perpendicular_dist
                perpendicular_dy /= perpendicular_dist

            # Move away from the enemy while still progressing towards the goal
            target_x = robot.x + perpendicular_dx * safe_distance + goal_dx * 0.25
            target_y = robot.y + perpendicular_dy * safe_distance + goal_dy * 0.25

    if goal_dist < 2:
        return None
    else:
        return target_x, target_y


def main():
    logging.basicConfig(level=logging.INFO)
    game = Game()
    stop_event = threading.Event()

    yellow_controller = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=1
    )
    blue_controller = RealRobotController(
        is_team_yellow=False, game_obj=game, n_robots=1
    )

    try:
        attacker_thread = threading.Thread(
            target=real_world_strat_atk, args=(game, yellow_controller, stop_event)
        )
        defender_thread = threading.Thread(
            target=real_world_strat_def, args=(game, blue_controller, stop_event)
        )

        attacker_thread.start()
        defender_thread.start()

        while attacker_thread.is_alive() or defender_thread.is_alive():
            time.sleep(0.1)
            if game.is_ball_in_goal() or ball_out_of_bounds(game.ball.x, game.ball.y):
                stop_event.set()
                break

    except KeyboardInterrupt:
        logger.info("Stopping due to keyboard interrupt")
        stop_event.set()
    finally:
        # Emergency stop
        for _ in range(15):
            yellow_controller.send_stop_command()
            blue_controller.send_stop_command()
        attacker_thread.join()
        defender_thread.join()
        logger.info("Robots stopped safely")


if __name__ == "__main__":
    main()
