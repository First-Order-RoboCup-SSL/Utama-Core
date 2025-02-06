import logging
import random
import math
import threading
import queue
import time
from tkinter import Y
from typing import Tuple

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.controllers.sim.grsim_robot_controller import (
    GRSimRobotController,
)
from team_controller.src.config.settings import TIMESTEP
from motion_planning.src.pid.pid import get_grsim_pids
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game
from entities.data.command import RobotCommand

# Imports from other scripts or modules within the same project
from robot_control.src.tests.utils import setup_pvp
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import (
    face_ball,
    go_to_point,
    velocity_to_orientation,
    clamp_to_goal_height,
    predict_goal_y_location,
)
from robot_control.src.intent import score_goal, find_likely_enemy_shooter, score_goal_

logger = logging.getLogger(__name__)

TOTAL_ITERATIONS = 10
MAX_GAME_TIME = 1000


def ball_out_of_bounds(ball_x: float, ball_y: float) -> bool:
    """
    Check if the ball is out of bounds.
    """
    return abs(ball_x) > 4.5 or abs(ball_y) > 3


def find_goal_position(
    game: Game,
) -> Tuple[float, float]:

    if game.my_team_is_yellow:
        goal_center = 4.5, 0
    else:
        goal_center = -4.5, 0
    goal_center = -4.5, 0

    # Assume that is_yellow <-> not is_left here # TODO : FIX
    _, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=game.my_team_is_yellow)
    shooters_data = find_likely_enemy_shooter(enemy, balls)

    orientation = None
    if not shooters_data:
        target_tracking_coord = game.ball.x, game.ball.y
        # TODO game.get_ball_velocity() can return (None, None)
        if (
            game.get_ball_velocity() is not None
            and None not in game.get_ball_velocity()
        ):
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
        predicted_goal_position = goal_center[0], clamp_to_goal_height(
            predict_goal_y_location(
                target_tracking_coord, orientation, game.my_team_is_yellow
            )
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
    grsim_controller: GRSimRobotController,
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
    goal_x, goal_y = -4.5, 0

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


def one_on_one(game: Game, stop_event: threading.Event, is_yellow: bool):
    """
    Worker function for the attacking strategy.
    """
    sim_robot_controller = GRSimRobotController(is_team_yellow=is_yellow)

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids(1)

    if game.my_team_is_yellow:
        goal_pos = (4.5, 0)  # Yellow team attacks the blue goal
    else:
        goal_pos = (-4.5, 0)  # Blue team attacks the yellow goal
    goal_pos = (-4.5, 0)

    goal_scored = False
    dribbling = False
    message = None

    dribble_task = DribbleToTarget(
        pid_oren,
        pid_trans,
        game,
        0,
        (0, 0),
        augment=True,
    )

    while not stop_event.is_set():
        # Process messages from the queue
        if not message_queue.empty():
            (message_type, message) = message_queue.get()
            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass

        if goal_scored:
            break

        if message is not None:
            if is_yellow:
                friendly_robot, enemy_robot = (
                    game.friendly_robots[0],
                    game.enemy_robots[0],
                )
            else:
                enemy_robot, friendly_robot = (
                    game.friendly_robots[0],
                    game.enemy_robots[0],
                )
            ball = game.ball

            enemy_dist_from_ball = math.hypot(
                ball.x - enemy_robot.x, ball.y - enemy_robot.y
            )
            friendly_dist_from_ball = math.hypot(
                ball.x - friendly_robot.x, ball.y - friendly_robot.y
            )

            # Check if the robot has the ball
            if sim_robot_controller.robot_has_ball(0):
                # Try to score
                cmd = score_goal_(
                    game,
                    True,
                    0,
                    pid_oren,
                    pid_trans,
                    True,
                    True,
                    is_yellow,
                )

                # If scoring is not possible, dribble to a better position
                if cmd is None:
                    if not dribbling:
                        # Calculate a target position for dribbling
                        target_coords = dribble_to_target_decision_maker(
                            game,
                            0,  # Robot ID
                            sim_robot_controller,
                            safe_distance=1.0,
                        )
                        if target_coords is not None:
                            dribble_task.update_coord(target_coords)
                            dribbling = True

                    # If dribbling, enact the dribble task
                    if dribbling:
                        pid_trans.max_velocity = 1.5
                        # print("Dribbling to target")
                        cmd = dribble_task.enact(sim_robot_controller.robot_has_ball(0))
                        # Check if the robot is close to the target
                        error = math.hypot(
                            target_coords[0] - friendly_robot.x,
                            target_coords[1] - friendly_robot.y,
                        )
                        if error < 0.1:  # Threshold for reaching the target
                            pid_trans.max_velocity = 2.5
                            dribbling = False
                else:
                    pid_trans.max_velocity = 2.5
                    dribbling = False  # Stop dribbling if scoring is possible
            elif dribbling:
                pid_trans.max_velocity = 1.5
                # print("Dribbling to target")
                cmd = dribble_task.enact(sim_robot_controller.robot_has_ball(0))
                # Check if the robot is close to the target
                error = math.hypot(
                    target_coords[0] - friendly_robot.x,
                    target_coords[1] - friendly_robot.y,
                )
                if error < 0.1:  # Threshold for reaching the target
                    pid_trans.max_velocity = 2.5
                    dribbling = False
            elif (
                enemy_dist_from_ball < 0.4
                and enemy_dist_from_ball < friendly_dist_from_ball
            ):
                cmd = improved_block_goal_and_attacker(
                    friendly_robot.robot_data,
                    game.enemy_robots[0].robot_data,
                    ball,
                    game,
                    pid_oren,
                    pid_trans,
                    attacker_has_ball=True,
                    block_ratio=0.4,
                    max_ball_follow_dist=1.0,
                )
            # elif friendly_dist_from_ball >= enemy_dist_from_ball:
            #     cmd = improved_block_goal_and_attacker(
            #         robot.robot_data,
            #         game.enemy_robots[0].robot_data,
            #         ball,
            #         game,
            #         pid_oren,
            #         pid_trans,
            #         attacker_has_ball=True,
            #         block_ratio=0.4,
            #         max_ball_follow_dist=1.0,
            #     )
            # elif enemy_dist_from_ball >= friendly_dist_from_ball:
            else:
                # If the robot doesn't have the ball, go to the ball
                cmd = go_to_point(
                    pid_oren,
                    pid_trans,
                    friendly_robot.robot_data,
                    friendly_robot.id,
                    (ball.x, ball.y),
                    face_ball((friendly_robot.x, friendly_robot.y), (ball.x, ball.y)),
                    dribbling=True,
                )

            # Send the command to the robot
            sim_robot_controller.add_robot_commands(cmd, 0)
            sim_robot_controller.send_robot_commands()

            # Check if a goal was scored
            if (
                game.is_ball_in_goal(right_goal=True)
                or game.is_ball_in_goal(right_goal=False)
                or ball_out_of_bounds(ball.x, ball.y)
            ):
                goal_scored = True
                break


def pvp_manager(headless: bool):
    """
    A 1v1 scenario with dynamic switching of attacker/defender roles.
    """
    # Initialize Game and environment
    env = GRSimController()

    env.reset()

    for i in range(1, 6):
        env.set_robot_presence(i, True, False)
    for i in range(1, 6):
        env.set_robot_presence(i, False, False)

    env.teleport_robot(True, 0, -1, -1.2)
    env.teleport_robot(False, 0, -1, 1.2)

    # ball_x = random.uniform(-1.5, 0)
    # ball_y = random.uniform(-1.5, 1.5)
    offset = random.uniform(-0.4, 0.4)
    env.teleport_ball(-1, offset)

    # Initialize Game objects for each team
    # yellow_game = Game(my_team_is_yellow=True)
    # blue_game = Game(my_team_is_yellow=False)
    game = Game(my_team_is_yellow=True)

    # Create a stop event to signal threads to stop
    stop_event = threading.Event()

    # Create threads for each team
    yellow_thread = threading.Thread(
        target=one_on_one,
        # target=strat_atk,
        # args=(yellow_game, stop_event),
        args=(game, stop_event, True),
    )
    blue_thread = threading.Thread(
        target=one_on_one,
        # target=strat_def,
        # args=(blue_game, stop_event),
        args=(game, stop_event, False),
    )

    yellow_thread.daemon = True
    blue_thread.daemon = True

    yellow_thread.start()
    blue_thread.start()

    time_elapsed = 0

    try:
        # Wait for threads to finish (they will run until a goal is scored or manually stopped)
        while yellow_thread.is_alive() or blue_thread.is_alive():
            time_elapsed += 1
            time.sleep(0.1)
            if time_elapsed > MAX_GAME_TIME:
                print("Game timed out.")
                stop_event.set()
                break
    except KeyboardInterrupt:
        print("Main thread interrupted. Stopping worker threads...")
        stop_event.set()  # Signal threads to stop
        TOTAL_ITERATIONS = -1

    # Wait for threads to finish
    yellow_thread.join()
    blue_thread.join()

    print("Game over.")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    try:
        for i in range(TOTAL_ITERATIONS):
            if TOTAL_ITERATIONS == -1:
                break
            pvp_manager(headless=False)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
