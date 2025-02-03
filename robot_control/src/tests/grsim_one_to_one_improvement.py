import logging
import random
import math
import threading
import queue
import time
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
from robot_control.src.skills import face_ball, go_to_point, go_to_ball
from robot_control.src.intent import score_goal

logger = logging.getLogger(__name__)


def improved_block_goal_and_attacker(
    robot,
    attacker,
    ball,
    goal_pos,
    pid_oren,
    pid_trans,
    attacker_has_ball: bool,
    block_ratio: float = 0.3,
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
        gx, gy = goal_pos[0], goal_pos[1]
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
    safe_distance: float = 1.0,
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
        goal_x, goal_y = 4.5, 0
    else:
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
            if enemy_robot is not None:
                enemy_dx = robot.x - enemy_robot.x
                enemy_dy = robot.y - enemy_robot.y
                enemy_dist = math.hypot(enemy_dx, enemy_dy)
                if smallest_enemy_dist == 0 or enemy_dist < smallest_enemy_dist:
                    smallest_enemy_dist = enemy_dist  
                    enemy_dx = enemy_dx
                    enemy_dy = enemy_dy    

        # If the enemy is too close, adjust the target position to avoid interception
        if enemy_dist < safe_distance:
            # Move perpendicular to the enemy-robot line to avoid the enemy
            perpendicular_dx = -enemy_dy
            perpendicular_dy = enemy_dx
            perpendicular_dist = math.hypot(perpendicular_dx, perpendicular_dy)

            # Normalize the perpendicular vector
            if perpendicular_dist > 0:
                perpendicular_dx /= perpendicular_dist
                perpendicular_dy /= perpendicular_dist

            # Move away from the enemy while still progressing towards the goal
            target_x = robot.x + perpendicular_dx * safe_distance + goal_dx * 0.5
            target_y = robot.y + perpendicular_dy * safe_distance + goal_dy * 0.5
        else:
            # Move directly towards the goal
            target_x = goal_x
            target_y = goal_y
    
    if goal_dist < 2:
        return None
    else:
        return target_x, target_y

def strat_a(game: Game, stop_event: threading.Event):
    """
    Worker function for the yellow team.
    """
    sim_robot_controller = GRSimRobotController(game.my_team_is_yellow)
    
    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids(1)  # Yellow
    
    if game.my_team_is_yellow:
        goal_pos = (4.5, 0)
    else:
        goal_pos = (-4.5, 0)
        
    goal_scored = False
    
    message = None
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
            robot = game.friendly_robots[0]
            ball = game.ball
            
            enemy_has_ball = False
            
            if sim_robot_controller.robot_has_ball(0):
                target_coords = dribble_to_target_decision_maker(
                        game,
                        robot.id,
                        sim_robot_controller,
                        safe_distance=1.0,
                    )
                cmd = DribbleToTarget(
                    pid_oren,
                    pid_oren,
                    game,
                    robot.id,
                    target_coords,
                    augment=True,
                )
                print(f"command: {cmd}")
            #     able_to_score = True
            #     if able_to_score: 
            #         cmd = score_goal(
            #             game,
            #             True,
            #             0,
            #             pid_oren,
            #             pid_trans,
            #             True,
            #             False,
            #         )
            #     else:
            #         target_coords = dribble_to_target_decision_maker(
            #             game,
            #             0,
            #             sim_robot_controller,
            #             safe_distance=1.0,
            #         )
            #         cmd = DribbleToTarget(
            #             pid_oren,
            #             pid_oren,
            #             game,
            #             0,
            #             target_coords,
            #             augment=True,
            #         )
            # elif enemy_has_ball:
            #     cmd = improved_block_goal_and_attacker(
            #         game.friendly_robots[0],
            #         game.enemy_robots[0],
            #         ball,
            #         goal_pos,
            #         pid_oren,
            #         pid_trans,
            #         attacker_has_ball=True,
            #         block_ratio=0.4,
            #         max_ball_follow_dist=1.0,
            #     )    
            else: 
                cmd = go_to_point(
                    pid_oren,
                    pid_trans,
                    robot.robot_data,
                    robot.id,
                    (ball.x, ball.y),
                    face_ball((robot.x, robot.y), (ball.x, ball.y)),
                )

            sim_robot_controller.add_robot_commands(cmd, 0)
            sim_robot_controller.send_robot_commands()
        
            # Check if goal was scored
            if game.is_ball_in_goal(our_side=True) or game.is_ball_in_goal(our_side=False):
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

    # Random ball placement
    ball_x = random.uniform(-2.5, 2.5)
    ball_y = random.uniform(-1.5, 1.5)
    env.teleport_ball(ball_x, ball_y)
    
    # Initialize Game objects for each team
    yellow_game = Game(my_team_is_yellow=True)
    blue_game = Game(my_team_is_yellow=False)

    # Create a stop event to signal threads to stop
    stop_event = threading.Event()

    # Create threads for each team
    yellow_thread = threading.Thread(
        target=strat_a,
        args=(yellow_game, stop_event),
    )
    blue_thread = threading.Thread(
        target=strat_a,
        args=(blue_game, stop_event),
    )

    yellow_thread.daemon = True
    blue_thread.daemon = True

    yellow_thread.start()
    blue_thread.start()
        
    try:
        # Wait for threads to finish (they will run until a goal is scored or manually stopped)
        while yellow_thread.is_alive() or blue_thread.is_alive():
            time.sleep(0.1)  # Avoid busy-waiting
    except KeyboardInterrupt:
        print("Main thread interrupted. Stopping worker threads...")
        stop_event.set()  # Signal threads to stop

    # Wait for threads to finish
    yellow_thread.join()
    blue_thread.join()

    print("Game over.")

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    try:
        pvp_manager(headless=False)
    except KeyboardInterrupt:
        print("Exiting...")
