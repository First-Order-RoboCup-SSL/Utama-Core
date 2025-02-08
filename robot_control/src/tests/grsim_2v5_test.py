import logging
import random
import math
import numpy as np
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

# from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game
from entities.data.command import RobotCommand

# Imports from other scripts or modules within the same project
from robot_control.src.tests.utils import setup_pvp
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import (
    face_ball,
    go_to_point,
    go_to_ball,
    empty_command,
    goalkeep,
    man_mark,
)
from robot_control.src.intent import score_goal, PassBall, defend_grsim
from global_utils.math_utils import distance
from robot_control.src.utils.pass_quality_utils import (
    find_pass_quality,
    find_best_receiver_position,
)
from robot_control.src.utils.shooting_utils import find_shot_quality

logger = logging.getLogger(__name__)

TOTAL_ITERATIONS = 1
MAX_GAME_TIME = 500
N_ROBOTS = 8
DEFENDING_ROBOTS = 6
ATTACKING_ROBOTS = 2
# TARGET_COORDS = (-2, 3)
PASS_QUALITY_THRESHOLD = 1.5
SHOT_QUALITY_THRESHOLD = 0.3

BALL_V0_MAGNITUDE = 3
BALL_A_MAGNITUDE = -0.3
START = 0


def ball_out_of_bounds(ball_x: float, ball_y: float) -> bool:
    """
    Check if the ball is out of bounds.
    """
    return abs(ball_x) > 4.5 or abs(ball_y) > 3


def intercept_ball(
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
    time_to_reach = (
        distance_to_ball / robot_speed if robot_speed != 0 else float("inf")
    )  # Assuming constant robot speed

    # Predict the future position of the ball
    intercept_pos = (
        (
            ball_pos[0] + ball_vel[0] * time_to_reach,
            ball_pos[1] + ball_vel[1] * time_to_reach,
        )
        if ball_vel != None
        else None
    )

    return intercept_pos


def attacker_strategy(game: Game, stop_event: threading.Event):
    """
    Worker function for the attacking strategy.
    """
    sim_robot_controller = GRSimRobotController(game.my_team_is_yellow)
    attacker_is_yellow = game.my_team_is_yellow
    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids(ATTACKING_ROBOTS)

    if game.my_team_is_yellow:
        goal_pos = (4.5, 0)  # Yellow team attacks the blue goal
    else:
        goal_pos = (-4.5, 0)  # Blue team attacks the yellow goal

    goal_scored = False
    message = None

    pass_task = None
    goal_scored = False
    trying_to_pass = False
    passer = None

    shoot_in_left_goal = game.my_team_is_yellow

    if shoot_in_left_goal:
        target_goal_line = game.field.LEFT_GOAL_LINE
    else:
        target_goal_line = game.field.RIGHT_GOAL_LINE
    friendly_robot_ids = [0, 1]

    # TODO
    player1_id = friendly_robot_ids[0]  # Start with robot 0
    player2_id = friendly_robot_ids[1]  # Start with robot 1

    # TODO: Not sure if this is sufficient for both blue and yellow scoring
    # It won't be because note that in real life the blue team is not necessarily
    # on the left of the pitch
    goal_x = target_goal_line.coords[0][0]
    goal_y1 = target_goal_line.coords[1][1]
    goal_y2 = target_goal_line.coords[0][1]

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
            commands = {}

            sampled_positions = None
            target_pos = None

            latest_frame = game.get_my_latest_frame(attacker_is_yellow)

            # if not latest_frame:
            #    return

            friendly_robots, enemy_robots, balls = latest_frame

            enemy_velocities = game.get_robots_velocity(attacker_is_yellow) or [
                (0.0, 0.0)
            ] * len(enemy_robots)

            if enemy_velocities is None or all(v is None for v in enemy_velocities):
                enemy_velocities = [(0.0, 0.0)] * len(enemy_robots)

            enemy_speeds = np.linalg.norm(enemy_velocities, axis=1)

            # TODO: For now we just look at the first ball, but this will eventually have to be smarter
            ball_data = balls[0]
            ball_vel = game.get_ball_velocity()

            player1_has_ball = sim_robot_controller.robot_has_ball(player1_id)
            player2_has_ball = sim_robot_controller.robot_has_ball(player2_id)
            ball_possessor_id = None

            if player1_has_ball:
                ball_possessor_id = player1_id
            elif player2_has_ball:
                ball_possessor_id = player2_id

            ### CASE 1: No one has the ball and we are not trying a pass - Try to intercept it ###
            if ball_possessor_id is None and trying_to_pass == False:
                print("CASEEEEE 1")
                print("No one has the ball, trying to intercept")
                best_interceptor = None
                best_intercept_score = float(
                    "inf"
                )  # Lower is better (closer to ball path)
                for rid in friendly_robot_ids:
                    ball_pos = ball_data.x, ball_data.y
                    robot = friendly_robots[rid]
                    # this is a bit hacky for now. we need a better interception function
                    # Calculate intercept position using the intercept_ball function
                    intercept_pos = intercept_ball(
                        (robot.x, robot.y), ball_pos, ball_vel, robot_speed=4.0
                    )  # Use appropriate robot speed

                    # Calculate how close the robot is to the intercept position (lower score is better)
                    intercept_score = (
                        distance(robot, intercept_pos)
                        if intercept_pos != None
                        else float("inf")
                    )

                    if intercept_score < best_intercept_score:
                        best_interceptor = rid
                        best_intercept_score = intercept_score

                # Send the best robot to intercept
                if best_interceptor is not None:
                    ball_vel = game.get_ball_velocity()
                    intercept_pos = intercept_ball(
                        (
                            friendly_robots[best_interceptor].x,
                            friendly_robots[best_interceptor].y,
                        ),
                        ball_pos,
                        ball_vel,
                        robot_speed=4.0,
                    )
                    commands[best_interceptor] = (
                        go_to_point(
                            pid_oren,
                            pid_trans,
                            friendly_robots[best_interceptor],
                            best_interceptor,
                            intercept_pos,
                            friendly_robots[best_interceptor].orientation,
                        )
                        if intercept_pos != None
                        else empty_command(dribbler_on=True)
                    )

                for rid in friendly_robot_ids:
                    if rid == best_interceptor or best_interceptor == None:
                        continue

                    potential_passer_id = (
                        rid + 1 if rid + 1 <= len(friendly_robots) - 1 else rid - 1
                    )
                    target_pos, sampled_positions, _ = find_best_receiver_position(
                        friendly_robots[rid],
                        friendly_robots[
                            potential_passer_id
                        ],  # PointOnField(ball_pos[0], ball_pos[1]),
                        enemy_robots,
                        enemy_speeds,
                        BALL_V0_MAGNITUDE,
                        BALL_A_MAGNITUDE,
                        goal_x,
                        goal_y1,
                        goal_y2,
                        attacker_is_yellow,
                    )

                    commands[rid] = go_to_ball(
                        pid_oren,
                        pid_trans,
                        friendly_robots[rid],
                        rid,
                        target_pos,
                        # friendly_robots[rid].orientation,
                    )

                # return commands, sampled_positions, target_pos

                sim_robot_controller.add_robot_commands(commands)
                print("intercepting")
                sim_robot_controller.send_robot_commands()

                continue

            ### CASE 2: Someone has the ball
            elif ball_possessor_id is not None and trying_to_pass == False:
                print("CASEEEEE 2")
                print("We have the ball", ball_possessor_id)
                possessor_data = friendly_robots[ball_possessor_id]

                # Check shot opportunity
                shot_quality = find_shot_quality(
                    possessor_data,
                    enemy_robots,
                    goal_x,
                    goal_y1,
                    goal_y2,
                    attacker_is_yellow,
                )
                print("the shot quality is ", shot_quality)
                if shot_quality > SHOT_QUALITY_THRESHOLD:
                    print("shooting with chance", shot_quality, SHOT_QUALITY_THRESHOLD)
                    commands[ball_possessor_id] = score_goal(
                        game,
                        True,
                        ball_possessor_id,
                        pid_oren,
                        pid_trans,
                        attacker_is_yellow,
                        attacker_is_yellow,
                    )
                    # return commands, None, None  # Just shoot, no need to pass
                    sim_robot_controller.add_robot_commands(commands)
                    sim_robot_controller.send_robot_commands()

                    continue

                # Check for best pass
                best_receiver_id = None
                best_pass_quality = 0

                for rid in friendly_robot_ids:
                    if rid == ball_possessor_id:
                        continue
                    pq = find_pass_quality(
                        friendly_robots[ball_possessor_id],
                        friendly_robots[rid],
                        enemy_robots,
                        enemy_speeds,
                        BALL_V0_MAGNITUDE,
                        BALL_A_MAGNITUDE,
                        goal_x,
                        goal_y1,
                        goal_y2,
                        attacker_is_yellow,
                    )
                    if pq > best_pass_quality:
                        best_pass_quality = pq
                        best_receiver_id = rid

                if (
                    best_receiver_id is not None
                    and best_pass_quality > PASS_QUALITY_THRESHOLD
                ) and pass_task is None:
                    print("made the passing thingy")
                    pass_task = PassBall(
                        pid_oren,
                        pid_trans,
                        game,
                        ball_possessor_id,
                        best_receiver_id,
                        (
                            friendly_robots[best_receiver_id].x,
                            friendly_robots[best_receiver_id].y,
                        ),
                    )
                end = time.time()
                START = end + 1 if START == None else START
                if (
                    best_receiver_id is not None
                    and best_pass_quality > PASS_QUALITY_THRESHOLD
                    and end - START >= 2
                ):
                    print(
                        "trying to execute a pass with quality ",
                        best_pass_quality,
                        PASS_QUALITY_THRESHOLD,
                    )
                    passer = ball_possessor_id
                    trying_to_pass = True
                    print("HELPPPPP we are in the first passing thingy")

                    pass_commands = pass_task.enact(passer_has_ball=True)
                    commands[ball_possessor_id] = pass_commands[0]
                    commands[best_receiver_id] = pass_commands[1]
                    # sim_robot_controller.add_robot_commands(commands)
                    # sim_robot_controller.send_robot_commands()
                    if sim_robot_controller.robot_has_ball(best_receiver_id):
                        print("pass finished")
                        trying_to_pass = False
                        pass_task = None
                else:
                    commands[ball_possessor_id] = empty_command(
                        dribbler_on=True
                    )  # Wait for a better pass

                # Move non-possessing robots to good positions
                for rid in friendly_robot_ids:
                    if rid == ball_possessor_id:
                        continue
                    # If a pass is happening, don't override the receiver's movement
                    if trying_to_pass:
                        continue  # Let PassBall handle the receiver
                    target_pos, sampled_positions, _ = find_best_receiver_position(
                        friendly_robots[rid],
                        friendly_robots[ball_possessor_id],
                        enemy_robots,
                        enemy_speeds,
                        BALL_V0_MAGNITUDE,
                        BALL_A_MAGNITUDE,
                        goal_x,
                        goal_y1,
                        goal_y2,
                        attacker_is_yellow,
                    )

                    commands[rid] = go_to_point(
                        pid_oren,
                        pid_trans,
                        friendly_robots[rid],
                        rid,
                        (target_pos.x, target_pos.y),
                        friendly_robots[rid].orientation,
                    )

                    # sim_robot_controller_attacker.send_robot_commands()
                    sim_robot_controller.add_robot_commands(commands)
                    sim_robot_controller.send_robot_commands()

            ### CASE 3: We are trying a pass ###
            elif trying_to_pass == True:
                print("CASEEEEE 3")
                """
                if pass_task == None:
                    print(made the ta)
                    pass_task = PassBall(
                        pid_oren,
                        pid_trans,
                        game,
                        ball_possessor_id,
                        best_receiver_id,
                        (
                            friendly_robots[best_receiver_id].x,
                            friendly_robots[best_receiver_id].y,
                        ),
                    )
                """
                print(passer, best_receiver_id)
                if best_receiver_id is None:
                    continue
                if trying_to_pass:
                    print(
                        "trying to execute a pass with quality ",
                        best_pass_quality,
                        PASS_QUALITY_THRESHOLD,
                    )
                    trying_to_pass = True
                    print("HELPPPPP we are in a new passing thingy")
                    assert pass_task is not None
                    pass_commands = pass_task.enact(passer_has_ball=True)
                    commands[passer] = pass_commands[0]
                    commands[best_receiver_id] = pass_commands[1]
                    # sim_robot_controller.add_robot_commands(commands)
                    # sim_robot_controller.send_robot_commands()
                    if sim_robot_controller.robot_has_ball(best_receiver_id):
                        print("pass finished")
                        trying_to_pass = False
                        passer = None
                        pass_task = None
                        START = time.time()
                else:
                    commands[passer] = empty_command(
                        dribbler_on=True
                    )  # Wait for a better pass

                sim_robot_controller.add_robot_commands(commands)
                sim_robot_controller.send_robot_commands()

            else:
                print("WTF is going on")

            # Check if a goal was scored
            if (
                game.is_ball_in_goal(our_side=True)
                or game.is_ball_in_goal(our_side=False)
                or ball_out_of_bounds(ball_data.x, ball_data.y)
            ):
                goal_scored = True
                break


def defender_strategy(game: Game, stop_event: threading.Event):
    sim_robot_controller = GRSimRobotController(game.my_team_is_yellow)
    my_team_is_yellow = game.my_team_is_yellow
    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids(DEFENDING_ROBOTS)

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
            _, _, balls = game.get_my_latest_frame(my_team_is_yellow)
            ball_data = balls[0]
            """
            sim_robot_controller.add_robot_commands(
                defend_grsim(
                    pid_oren,
                    pid_trans,
                    game,
                    my_team_is_yellow,
                    1,
                ),
                1,
            )
            """
            """
            sim_robot_controller.add_robot_commands(
                defend_grsim(
                    pid_oren,
                    pid_trans,
                    game,
                    my_team_is_yellow,
                    4,
                ),
                4,
            )
            """
            sim_robot_controller.add_robot_commands(
                goalkeep(
                    not my_team_is_yellow,
                    game,
                    0,
                    pid_oren,
                    pid_trans,
                    my_team_is_yellow,
                    sim_robot_controller.robot_has_ball(0),
                ),
                0,
            )
            """
            sim_robot_controller.add_robot_commands(
                man_mark(
                    my_team_is_yellow,
                    game,
                    2,
                    0,
                    pid_oren,
                    pid_trans,
                ),
                2,
            )

            sim_robot_controller.add_robot_commands(
                man_mark(
                    my_team_is_yellow,
                    game,
                    3,
                    0,
                    pid_oren,
                    pid_trans,
                ),
                3,
            )
            """
            sim_robot_controller.send_robot_commands()

            # Check if a goal was scored
            if (
                game.is_ball_in_goal(our_side=True)
                or game.is_ball_in_goal(our_side=False)
                or ball_out_of_bounds(ball_data.x, ball_data.y)
            ):
                goal_scored = True
                break


def pvp_manager(headless: bool):
    # Initialize Game and environment
    env = GRSimController()

    env.reset()

    for i in range(ATTACKING_ROBOTS, 6):
        # yellow team
        env.set_robot_presence(i, True, False)
    for i in range(DEFENDING_ROBOTS, 6):
        # blue team
        env.set_robot_presence(i, False, False)

    # Random ball placement
    ball_x = random.uniform(-2.5, 2.5)
    ball_y = random.uniform(-1.5, 1.5)
    # env.teleport_ball(0.5, 0)
    env.teleport_ball(ball_x, ball_y)

    # Initialize Game objects for each team
    yellow_game = Game(my_team_is_yellow=True)
    blue_game = Game(my_team_is_yellow=False)

    # yellow_robs = yellow_game.get_my_latest_frame(True)
    # print(yellow_robs)

    # Create a stop event to signal threads to stop
    stop_event = threading.Event()

    # Create threads for each team
    yellow_thread = threading.Thread(
        target=attacker_strategy,
        args=(yellow_game, stop_event),
    )
    blue_thread = threading.Thread(
        target=defender_strategy,
        args=(blue_game, stop_event),
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

    # Wait for threads to finish
    yellow_thread.join()
    blue_thread.join()

    print("Game over.")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    try:
        for i in range(TOTAL_ITERATIONS):
            pvp_manager(headless=False)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
