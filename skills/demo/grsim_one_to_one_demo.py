import logging
import math
import queue
import random
import sys
import threading
import time
from typing import Tuple

from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import find_likely_enemy_shooter, score_goal
from robot_control.src.skills import (
    clamp_to_goal_height,
    go_to_ball,
    go_to_point,
    predict_goal_y_location,
    velocity_to_orientation,
)

from entities.data.command import RobotCommand
from entities.game import Game
from motion_planning.src.pid.pid import get_grsim_pids
from team_controller.src.controllers.real.real_robot_controller import (
    RealRobotController,
)
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.controllers.sim.grsim_robot_controller import (
    GRSimRobotController,
)
from team_controller.src.data import VisionReceiver
from team_controller.src.data.message_enum import MessageType

# Imports from other scripts or modules within the same project


logger = logging.getLogger(__name__)

TOTAL_ITERATIONS = 100
MAX_GAME_TIME = 10000
SHOOT_AT_BLUE_GOAL = True
CLOSE = False
# Shoot at blue goal means shooting to the left in rsim environement
FIELD_X_MIN, FIELD_X_MAX = -4.5, 0
FIELD_Y_MIN, FIELD_Y_MAX = -3, 3


def ball_out_of_bounds(ball_x: float, ball_y: float) -> bool:
    """Check if the ball is out of bounds."""
    return 0 < ball_x < -4.25 or abs(ball_y) > 3


def find_goal_position(
    game: Game,
) -> Tuple[float, float]:
    goal_center = (-4.5, 0) if SHOOT_AT_BLUE_GOAL else (4.5, 0)

    enemies = []
    for enemy in game.enemy_robots:
        enemies.append(enemy.robot_data)
    shooters_data = find_likely_enemy_shooter(enemies, [game.ball])

    orientation = None
    if not shooters_data:
        target_tracking_coord = game.ball.x, game.ball.y
        # TODO game.get_ball_velocity() can return (None, None)
        if game.get_ball_velocity() is not None and None not in game.get_ball_velocity():
            orientation = velocity_to_orientation(game.get_ball_velocity())
    else:
        # TODO (deploy more defenders, or find closest shooter?)
        sd = shooters_data[0]
        target_tracking_coord = sd.x, sd.y
        orientation = sd.orientation

    if orientation is None:
        # In case there is no ball velocity or attackers, use centre of goal
        predicted_goal_position = goal_center
    else:
        predicted_goal_position = (
            goal_center[0],
            clamp_to_goal_height(predict_goal_y_location(target_tracking_coord, orientation, game.my_team_is_yellow)),
        )

    return predicted_goal_position


def improved_block_goal_and_attacker(
    robot,
    attacker,
    ball,
    game: Game,
    pid_oren,
    pid_trans,
    attacker_has_ball: bool,
    block_ratio: float = 0.1,
    max_ball_follow_dist: float = 1.0,
) -> RobotCommand:
    """
    Intelligent defense strategy:
    1) If the attacker has the ball, block on the attacker-goal line.
    2) Otherwise, stay closer to the ball while still considering the attacker's possible shot.

    :param robot: Defender's Robot object (robot.x, robot.y, robot.orientation)
    :param attacker: Attacker's Robot object
    :param ball: Ball object
    :param goal_pos: The (x, y) location of the goal to defend
    :param pid_oren: PID controller for orientation
    :param pid_trans: PID controller for translation
    :param attacker_has_ball: Whether the attacker currently has ball possession
    :param block_ratio: 0~1 ratio to position ourselves between attacker & goal
    :param max_ball_follow_dist: if attacker doesn't have ball, how close we stay near the ball
    :return: The command dict for the defender robot
    """
    if attacker_has_ball:
        # ========== Prioritize blocking the shot line ==========
        ax, ay = attacker.x, attacker.y

        gx, gy = find_goal_position(game)
        # print(gx, gy)

        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            # Extreme edge case if attacker and goal are basically the same
            target_x, target_y = gx, gy
        else:
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy

        # Face the attacker
        face_theta = math.atan2((ay - robot.y), (ax - robot.x))
    else:
        # ========== Attacker doesn't have ball; defend ball more closely ==========
        ax, ay = attacker.x, attacker.y
        bx, by = ball.x, ball.y

        # Move ~70% of the way toward the ball from the attacker
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        # If we are too far from the ball, move closer
        dist_def_to_ball = math.hypot(robot.x - bx, robot.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            ratio = max_ball_follow_dist / dist_def_to_ball
            target_x = robot.x + (bx - robot.x) * ratio
            target_y = robot.y + (by - robot.y) * ratio

        # Face the ball
        face_theta = math.atan2((by - robot.y), (bx - robot.x))

    # === IMPORTANT CHANGE HERE ===
    # Instead of passing the Robot object directly to go_to_point(),
    # we pass (x, y, orientation) as a tuple.
    current_pose = (
        robot.x,
        robot.y,
        robot.orientation if hasattr(robot, "orientation") else 0.0,
    )

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        current_pose,  # (x, y, orientation) tuple
        0,
        (target_x, target_y),
        face_theta,
    )
    return cmd


def dribble_to_target_decision_maker(
    game: Game,
    robot_id: int,
    real_controller: RealRobotController,
    safe_distance: float = 0.2,
) -> Tuple[float, float]:
    """Determines the optimal (x, y) position for the robot to dribble to, avoiding interception by the enemy.

    :param game: The game object containing current state information.
    :param robot_id: The ID of the robot making the decision.
    :param goal_pos: The (x, y) position of the goal to score in.
    :param enemy_robot_pos: The (x, y) position of the enemy robot.
    :param ball_pos: The (x, y) position of the ball.
    :param safe_distance: The minimum distance to maintain from the enemy robot.
    :return: The optimal (x, y) position for the robot to dribble to.
    """
    robot = game.friendly_robots[robot_id]  # Assuming the robot is on the yellow team

    goal_x, goal_y = -4.5 if SHOOT_AT_BLUE_GOAL else 4.5, 0

    # If the robot has the ball, move towards the goal while avoiding the enemy
    if real_controller.robot_has_ball(robot_id):
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
            print("Safe distance")
            target_x = robot.x + perpendicular_dx + goal_dx * 0.2
            target_y = robot.y + perpendicular_dy + goal_dy * 0.2
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

            print("Not safe distance")
            # Move away from the enemy while still progressing towards the goal
            target_x = robot.x + perpendicular_dx * 0.6 + goal_dx * 0.4
            target_y = robot.y + perpendicular_dy * 0.4 + goal_dy * 0.6

        if goal_dist < 2:
            return None
        else:
            return target_x, target_y


def one_on_one(
    game: Game,
    stop_event: threading.Event,
    friendly_robot_id: int,
    enemy_robot_id: int,
    robot_controller: RealRobotController = None,
):
    robot_controller = GRSimRobotController(is_team_yellow=game.my_team_is_yellow)

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionReceiver(message_queue, n_cameras=4)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids()

    goal_scored = False
    dribbling = False
    message = None
    target_coords = None
    can_score = False

    dribble_task = DribbleToTarget(
        pid_oren,
        pid_trans,
        game,
        0,
        (0, 0),
        augment=False,
    )

    while not stop_event.is_set():
        # Process messages from the queue
        if not message_queue.empty():
            (message_type, message) = message_queue.get()
            if message_type == MessageType.VISION:
                # print(message)
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass

        if goal_scored:
            break

        if message is not None:
            friendly_robot, enemy_robot = (
                game.friendly_robots[friendly_robot_id],
                game.enemy_robots[enemy_robot_id],
            )
            ball = game.ball

            enemy_dist_from_ball = math.hypot(ball.x - enemy_robot.x, ball.y - enemy_robot.y)
            friendly_dist_from_ball = math.hypot(ball.x - friendly_robot.x, ball.y - friendly_robot.y)

            if robot_controller.robot_has_ball(0):
                cmd = score_goal(
                    game,
                    True,
                    friendly_robot_id,
                    pid_oren,
                    pid_trans,
                )
                if cmd is None:
                    can_score = False
                    if not dribbling:
                        # Calculate a target position for dribbling
                        target_coords = dribble_to_target_decision_maker(
                            game,
                            friendly_robot_id,  # Robot ID
                            robot_controller,
                            safe_distance=1.0,
                        )
                        if target_coords:
                            dribble_task.update_coord(target_coords)
                            dribbling = True
                    else:
                        # print(f"Yellow: {game.my_team_is_yellow}, Dribbling")
                        # print("Dribbling to target")
                        cmd = dribble_task.enact(robot_controller.robot_has_ball(0))
                        # Check if the robot is close to the target
                        error = math.hypot(
                            target_coords[0] - friendly_robot.x,
                            target_coords[1] - friendly_robot.y,
                        )
                        if error < 0.08:  # Threshold for reaching the target
                            dribbling = False
                else:
                    can_score = True
            elif enemy_dist_from_ball <= friendly_dist_from_ball and not dribbling:
                cmd = improved_block_goal_and_attacker(
                    friendly_robot.robot_data,
                    enemy_robot.robot_data,
                    ball,
                    game,
                    pid_oren,
                    pid_trans,
                    attacker_has_ball=True,
                    block_ratio=0.4,
                    max_ball_follow_dist=1.0,
                )
            elif not dribbling:
                cmd = go_to_ball(pid_oren, pid_trans, friendly_robot.robot_data, 0, ball)

            # If dribbling, enact the dribble task
            if dribbling and not can_score:
                # print(f"Yellow: {game.my_team_is_yellow}, Dribbling")
                # print("Dribbling to target")
                cmd = dribble_task.enact(robot_controller.robot_has_ball(0))
                # Check if the robot is close to the target
                error = math.hypot(
                    target_coords[0] - friendly_robot.x,
                    target_coords[1] - friendly_robot.y,
                )
                if error < 0.08:  # Threshold for reaching the target
                    dribbling = False
            elif can_score and dribbling and not robot_controller.robot_has_ball(0):
                dribbling = False
                cmd = go_to_ball(pid_oren, pid_trans, friendly_robot.robot_data, 0, ball)

            # if robot_controller.robot_has_ball(0):
            #     print(f"Yellow: {game.my_team_is_yellow}, Has Ball")
            # else:
            #     print(f"Yellow: {game.my_team_is_yellow}, No Ball")
            # if dribbling:
            #     print(f"Yellow: {game.my_team_is_yellow}, Dribbling")
            # else:
            #     print(f"Yellow: {game.my_team_is_yellow}, Not Dribbling")
            # if can_score:
            #     print(f"Yellow: {game.my_team_is_yellow}, Scoring")
            #     print(cmd)
            # else:
            #     print(f"Yellow: {game.my_team_is_yellow}, Not Scoring")

            # Send the command to the robot
            robot_controller.add_robot_commands(cmd, friendly_robot_id)
            robot_controller.send_robot_commands()

            # Check if a goal was scored
            if (
                game.is_ball_in_goal(right_goal=True)
                or game.is_ball_in_goal(right_goal=False)
                or ball_out_of_bounds(ball.x, ball.y)
            ):
                goal_scored = True
                break


def one_to_one_sim(headless: bool):
    """A 1v1 scenario with dynamic switching of attacker/defender roles."""

    # Initialize Game and environment
    env = GRSimController()
    env.reset()

    time.sleep(0.01)

    for i in range(1, 6):
        env.set_robot_presence(i, is_team_yellow=True, is_present=False)
    for i in range(1, 6):
        env.set_robot_presence(i, is_team_yellow=False, is_present=False)

    time.sleep(0.1)

    # Teleport robot
    env.teleport_robot(True, 0, -1, -1.2)
    env.teleport_robot(False, 0, -1, 1.2)

    time.sleep(0.1)

    # Teleport ball
    offset = random.uniform(-0.3, 0.3)
    env.teleport_ball(-1.5, offset)

    # Initialize Game objects for each team
    yellow_game = Game(my_team_is_yellow=True)
    blue_game = Game(my_team_is_yellow=False)

    # Create a stop event to signal threads to stop
    stop_event = threading.Event()

    # robot_controller = RealRobotController(
    #     is_team_yellow=True, game_obj=yellow_game, n_robots=5
    # )

    # Create threads for each team
    yellow_thread = threading.Thread(
        target=one_on_one,
        args=(yellow_game, stop_event, 0, 0),
    )
    blue_thread = threading.Thread(
        target=one_on_one,
        args=(blue_game, stop_event, 0, 0),
    )

    yellow_thread.daemon = True
    blue_thread.daemon = True

    yellow_thread.start()
    blue_thread.start()

    time_elapsed = 0

    try:
        # Wait for threads to finish (they will run until a goal is scored or manually stopped)
        while yellow_thread.is_alive() and blue_thread.is_alive():
            time_elapsed += 1
            time.sleep(0.1)
            if time_elapsed > MAX_GAME_TIME:
                print("Game timed out.")
                stop_event.set()
                break
    except KeyboardInterrupt:
        print("Main thread interrupted. Stopping worker threads...")  #
        # for i in range(8): # Try really hard to stop the robots!
        #     robot_controller.serial.write(stop_buffer_off)
        stop_event.set()  # Signal threads to stop
        raise

    # Wait for threads to finish
    # yellow_thread.join()
    blue_thread.join()

    print("Game over.")


if __name__ == "__main__":
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]
    logging.disable(logging.WARNING)
    try:
        for i in range(TOTAL_ITERATIONS):
            one_to_one_sim(headless=False)  # This will now propagate the exception
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)  # Ensures the program exits cleanly
