import logging
import math
import queue
import random
import threading
import time

from robot_control.src.intent import PassBall, defend, score_goal
from robot_control.src.skills import empty_command, go_to_ball, go_to_point, goalkeep
from robot_control.src.utils.shooting_utils import find_best_shot
from vision.vision_receiver import VisionReceiver

from entities.game import Game
from motion_planning.src.pid.pid import get_grsim_pids
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.controllers.sim.grsim_robot_controller import (
    GRSimRobotController,
)
from team_controller.src.data.message_enum import MessageType

random.seed(15)

MAX_GAME_TIME = 500
TOTAL_ITERATIONS = 1
N_ROBOTS_ATTACK = 3
N_ROBOTS_DEFEND = 2

START_POS = -2
SPACING_Y = 1.5
SPACING_X = 1
SPAWN_BOX_SIZE = 1


def defender_strategy(game: Game, stop_event: threading.Event):
    sim_robot_controller_defender = GRSimRobotController(game.my_team_is_yellow)
    my_team_is_yellow = game.my_team_is_yellow
    message_queue = queue.SimpleQueue()

    vision_receiver = VisionReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren_defender, pid_2d_defender = get_grsim_pids(N_ROBOTS_DEFEND)
    pid_2d_defender.dimX.Kp = 1
    pid_2d_defender.dimY.Kp = 1

    message = None
    while not stop_event.is_set():
        # Process messages from the queue
        # start = time.time()
        if not message_queue.empty():
            (message_type, message) = message_queue.get()
            if message_type == MessageType.VISION:
                game.add_new_state(message)

                defender_command = defend(pid_oren_defender, pid_2d_defender, game, my_team_is_yellow, 1, None)
                goalie_command = goalkeep(
                    not my_team_is_yellow,
                    game,
                    0,
                    pid_oren_defender,
                    pid_2d_defender,
                    my_team_is_yellow,
                    sim_robot_controller_defender.robot_has_ball(0),
                )
                # # goalie_command = go_to_point(pid_oren_defender, pid_2d_defender, game.get_robot_pos(False, 0), 0, (4.5, 0), False)
                sim_robot_controller_defender.add_robot_commands({1: defender_command, 0: goalie_command})
                sim_robot_controller_defender.send_robot_commands()

            elif message_type == MessageType.REF:
                pass


def attacker_strategy(game: Game, stop_event: threading.Event):
    sim_robot_controller_attacker = GRSimRobotController(game.my_team_is_yellow)
    message_queue = queue.SimpleQueue()
    vision_receiver = VisionReceiver(message_queue)
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren_attacker, pid_2d_attacker = get_grsim_pids(N_ROBOTS_ATTACK)

    if game.my_team_is_yellow:
        target_pos = [(START_POS - (i + 1) % 2 * SPACING_X, SPACING_Y - SPACING_Y * i) for i in range(N_ROBOTS_ATTACK)]
    else:
        target_pos = [(-START_POS + (i + 1) % 2 * SPACING_X, SPACING_Y - SPACING_Y * i) for i in range(N_ROBOTS_ATTACK)]

    pass_task = None
    # Never used !!!
    # shooting = False
    # goal_scored = False
    stage = 0
    passes = 0
    message = None
    iter = 0
    while not stop_event.is_set():
        # Process messages from the queue
        if not message_queue.empty():
            (message_type, message) = message_queue.get()
            iter += 1

            if message_type == MessageType.VISION:
                game.add_new_state(message)

            if stage == 0:
                if iter == 10:  # give them chance to spawn in the correct place
                    stage += 1

            elif stage == 1:
                closest_robot = None
                closest_distance = float("inf")
                for i in range(N_ROBOTS_ATTACK):
                    robot_data = game.get_robot_pos(game.my_team_is_yellow, i)
                    ball = game.get_ball_pos()[0]

                    distance = math.dist((robot_data.x, robot_data.y), (ball.x, ball.y))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_robot = i

                all_in_pos = True
                for i in range(N_ROBOTS_ATTACK):
                    robot_data = game.get_robot_pos(game.my_team_is_yellow, i)
                    ball = game.get_ball_pos()[0]

                    if i == closest_robot and not sim_robot_controller_attacker.robot_has_ball(i):
                        sim_robot_controller_attacker.add_robot_commands(
                            go_to_ball(pid_oren_attacker, pid_2d_attacker, robot_data, i, ball),
                            i,
                        )
                        possessor = i
                    else:
                        sim_robot_controller_attacker.add_robot_commands(
                            go_to_point(
                                pid_oren_attacker,
                                pid_2d_attacker,
                                robot_data,
                                i,
                                target_pos[i],
                                math.pi,
                                True,
                            ),
                            i,
                        )

                    all_in_pos = all_in_pos and math.dist(target_pos[i], (robot_data.x, robot_data.y)) < 0.01
                sim_robot_controller_attacker.send_robot_commands()

                if all_in_pos:
                    stage += 1

            elif stage == 2:
                if not pass_task:
                    target_goal_line = game.field.enemy_goal_line(game.my_team_is_yellow)
                    latest_frame = game.get_my_latest_frame(game.my_team_is_yellow)
                    if latest_frame:
                        friendly_robots, enemy_robots, balls = latest_frame

                        goal_x = target_goal_line.coords[0][0]
                        goal_y1 = target_goal_line.coords[1][1]
                        goal_y2 = target_goal_line.coords[0][1]

                        best_shot, size_of_shot = find_best_shot(
                            balls[0],
                            enemy_robots,
                            goal_x,
                            goal_y1,
                            goal_y2,
                            game.my_team_is_yellow,
                        )

                        if size_of_shot > 0.41 and passes >= 5:
                            stage += 1

                    passes += 1
                    if possessor == 0:
                        next_possessor = 1
                    elif possessor == N_ROBOTS_ATTACK - 1:
                        next_possessor = N_ROBOTS_ATTACK - 2
                    else:
                        next_possessor = random.choice([possessor + 1, possessor - 1])

                    pass_task = PassBall(
                        pid_oren_attacker,
                        pid_2d_attacker,
                        game,
                        possessor,
                        next_possessor,
                        target_coords=game.get_robot_pos(game.my_team_is_yellow, next_possessor),
                    )
                # else:
                if sim_robot_controller_attacker.robot_has_ball(next_possessor):
                    pass_task = None
                    possessor = next_possessor
                    sim_robot_controller_attacker.add_robot_commands(empty_command(dribbler_on=True), possessor)
                    sim_robot_controller_attacker.add_robot_commands(empty_command(dribbler_on=True), 0)
                    sim_robot_controller_attacker.send_robot_commands()
                else:
                    (possessor_cmd, next_possessor_cmd) = pass_task.enact(
                        sim_robot_controller_attacker.robot_has_ball(possessor)
                    )
                    sim_robot_controller_attacker.add_robot_commands(possessor_cmd, possessor)
                    sim_robot_controller_attacker.add_robot_commands(next_possessor_cmd, next_possessor)
                    sim_robot_controller_attacker.send_robot_commands()
            elif stage == 3:
                sim_robot_controller_attacker.add_robot_commands(
                    score_goal(
                        game,
                        sim_robot_controller_attacker.robot_has_ball(possessor),
                        possessor,
                        pid_oren_attacker,
                        pid_2d_attacker,
                        game.my_team_is_yellow,
                        game.my_team_is_yellow,
                    ),
                    possessor,
                )
                sim_robot_controller_attacker.send_robot_commands()
            elif message_type == MessageType.REF:
                pass


def pvp_manager(headless: bool, attacker_is_yellow: bool):
    """A 1v1 scenario with dynamic switching of attacker/defender roles."""
    # Initialize Game and environment
    env = GRSimController()

    env.reset()

    env.teleport_ball(random.uniform(-3, 3), random.uniform(-3, 3))

    for i in range(N_ROBOTS_ATTACK):
        # print(random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE))
        # env.teleport_robot(attacker_is_yellow, i, target_pos[i][0] + random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE), target_pos[i][1] + random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE))
        env.teleport_robot(attacker_is_yellow, i, 0, -0.3 * (i - 1))

    for i in range(N_ROBOTS_ATTACK, 6):
        # yellow team
        env.set_robot_presence(i, True, False)
    for i in range(N_ROBOTS_DEFEND, 6):
        # blue team
        env.set_robot_presence(i, False, False)

    # Random ball placement
    ball_x = random.uniform(-2.5, 2.5)
    ball_y = random.uniform(-1.5, 1.5)
    env.teleport_ball(ball_x, ball_y)
    # env.teleport_ball(1, 1)

    # Initialize Game objects for each team
    yellow_game = Game(my_team_is_yellow=True)
    blue_game = Game(my_team_is_yellow=False)

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
        time.sleep(5)
        for i in range(TOTAL_ITERATIONS):
            pvp_manager(headless=False, attacker_is_yellow=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
