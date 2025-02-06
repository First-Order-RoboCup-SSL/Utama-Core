from calendar import c
from motion_planning.src.pid.pid import get_real_pids
from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import score_goal
from robot_control.src.skills import go_to_point, go_to_ball, turn_on_spot
from team_controller.src.controllers import RealRobotController
from entities.game import Game
from entities.data.command import RobotCommand
import time
import queue
import logging
import threading
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data.vision_receiver import VisionDataReceiver
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

random.seed(10)

# Need to tune time frames
# Need to import goalkeep

def one_vs_one_goalie(game: Game, robot_controller_goalie: RealRobotController, robot_controller_attacker: RealRobotController):
    ATTACKER_ROBOT_ID = 1
    GOALIE_ROBOT_ID = 1
    ATTACKER_IS_YELLOW = True
    LEFT_GOAL = True

    pid_oren_goalie, pid_trans_goalie = get_real_pids(6)
    pid_oren_attacker, pid_trans_attacker = get_real_pids(6)

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    start_time = time.time()

    dribble_task = None
    current_x = -0.5

    while not goal_scored and time.time() - start_time < 15:
        if game.is_ball_in_goal(our_side=True):
            logger.info("Goal Scored at Position: ", game.get_ball_pos())
            goal_scored = True
        
        (message_type, message) = message_queue.get()  # Infinite timeout for now

        if message_type == MessageType.VISION:
            game.add_new_state(message)


            if time.time() - start_time < 3:
                robot_controller_attacker.add_robot_commands(go_to_ball(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), ATTACKER_ROBOT_ID, game.ball), ATTACKER_ROBOT_ID)    
                robot_controller_goalie.add_robot_commands(go_to_point(pid_oren_goalie, pid_trans_goalie, game.get_robot_pos(not ATTACKER_IS_YELLOW, GOALIE_ROBOT_ID), 1, (-4.5, 0), GOALIE_ROBOT_ID, False), GOALIE_ROBOT_ID)
                robot_controller_attacker.send_robot_commands()
                robot_controller_goalie.send_robot_commands()
            elif time.time() - start_time < 5:
                robot_controller_attacker.add_robot_commands(go_to_point(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), 1, (-0.5, 0), ATTACKER_ROBOT_ID, dribbling=True), ATTACKER_ROBOT_ID)
            elif time.time() - start_time < 7:
                if not dribble_task and current_x > -3:
                    current_x -= 1
                    dribble_task = DribbleToTarget(pid_oren_attacker, pid_trans_attacker, game, ATTACKER_ROBOT_ID, (current_x, random.uniform(-1, 1)))
                    robot_controller_attacker.add_robot_commands(dribble_task.enact())
                    robot_controller_attacker.send_robot_commands()
            else:
                robot_controller_goalie.add_robot_commands(goalkeep(not ATTACKER_IS_YELLOW, game, GOALIE_ROBOT_ID, pid_oren_goalie, pid_trans_goalie, False, False))
                robot_controller_attacker.add_robot_commands(score_goal(game, shooter_has_ball=True, robot_id=ATTACKER_ROBOT_ID, pid_oren=pid_oren_attacker, pid_trans=pid_trans_attacker, is_yellow=ATTACKER_IS_YELLOW, shoot_in_left_goal=LEFT_GOAL))
                robot_controller_goalie.send_robot_commands()
                robot_controller_attacker.send_robot_commands()

def main():
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]

    game = Game()
    robot_controller_yellow = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=1
    )
    robot_controller_blue = RealRobotController(
        is_team_yellow=False, game_obj=game, n_robots=1
    )

    try:
        one_vs_one_goalie(game, robot_controller_yellow, robot_controller_blue)
    except KeyboardInterrupt:
        print("Stopping robot.")
    finally:
        for i in range(15): # Try really hard to stop the robots!
            robot_controller_yellow.serial.write(stop_buffer_off)
            robot_controller_blue.serial.write(stop_buffer_off)

if __name__ == "__main__":
    main()
