from calendar import c
from motion_planning.src.pid.pid import get_real_pids, get_real_pids_goalie
from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import score_goal
from robot_control.src.skills import go_to_point, go_to_ball, goalkeep, turn_on_spot
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
# from mock import Mock
import math

logger = logging.getLogger(__name__)

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

random.seed(10)


def one_vs_one_goalie(game: Game, robot_controller_attacker: RealRobotController, robot_controller_goalie: RealRobotController):
    ATTACKER_ROBOT_ID = 1 # 1
    GOALIE_ROBOT_ID = 4 # 4
    ATTACKER_IS_YELLOW = True
    LEFT_GOAL = True

    pid_oren_goalie, pid_trans_goalie = get_real_pids_goalie(6)
    pid_oren_attacker, pid_trans_attacker = get_real_pids(6)

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    # dribble_task = None
    current_x = -0.5
    goal_scored = False

    stage = 1
    iter = 0
    ball_init_pos = None

    while not goal_scored:
        iter += 1
        (message_type, message) = message_queue.get()
        if message_type == MessageType.VISION:
            game.add_new_state(message)

            if iter == 2:
                ball_init_pos = game.get_ball_pos()[0]

            if game.is_ball_in_goal(right_goal=False):
                logger.info("Goal Scored at Position: ", game.get_ball_pos())
                goal_scored = True

            # Go to point
            # game.get_robot_pos(not ATTACKER_IS_YELLOW, GOALIE_ROBOT_ID)
            # robot_controller_goalie.add_robot_commands(go_to_point(pid_oren_goalie, pid_trans_goalie, game.get_robot_pos(not ATTACKER_IS_YELLOW, GOALIE_ROBOT_ID), GOALIE_ROBOT_ID, (-3.5, 0), 0, False), GOALIE_ROBOT_ID)
            
            # Keeper only
            # robot_controller_goalie.add_robot_commands(goalkeep(is_left_goal=True, game=game, robot_id=GOALIE_ROBOT_ID, pid_oren=pid_oren_goalie, pid_trans=pid_trans_goalie, is_yellow=not ATTACKER_IS_YELLOW, goalie_has_ball=False), GOALIE_ROBOT_ID)
            # robot_controller_goalie.send_robot_commands()

            # # Full play
            new_ball_pos = game.get_ball_pos()[0]
            attacker_pos = game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID)
            
            robot_controller_goalie.add_robot_commands(goalkeep(True, game, GOALIE_ROBOT_ID, pid_oren_goalie, pid_trans_goalie, not ATTACKER_IS_YELLOW, False), GOALIE_ROBOT_ID)

            if ball_init_pos and math.dist((new_ball_pos.x, new_ball_pos.y), (ball_init_pos.x, ball_init_pos.y)) > 0.1 and stage == 1: # and math.dist((goalie_pos.x, goalie_pos.y), (-4.5, 0)) 
                stage += 1
            elif math.dist((attacker_pos.x, attacker_pos.y), (-3, 0.5)) < 0.07 and stage == 2:
                stage += 1
                target = (current_x, random.uniform(-1, 1))
            # elif stage == 3: # target is not None and math.dist((attacker_pos.x, attacker_pos.y), target) < 0.2 and 
            #     stage += 1
            
            print(f"STAGE {stage}")

            if stage == 1:  
                print("STAGE 1", ball_init_pos, new_ball_pos)  
                robot_controller_attacker.add_robot_commands(go_to_ball(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), ATTACKER_ROBOT_ID, game.ball), ATTACKER_ROBOT_ID)    
            elif stage == 2:
                print("STAGE 2")
                robot_controller_attacker.add_robot_commands(go_to_point(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), 1, (-3, 0.5), None, dribbling=True), ATTACKER_ROBOT_ID)
            # elif stage == 3:
            #     print("STAGE 3")
            #     if not dribble_task and current_x > -3:
            #         current_x -= 1
            #         dribble_task = DribbleToTarget(pid_oren_attacker, pid_trans_attacker, game, ATTACKER_ROBOT_ID, target)
            #         robot_controller_attacker.add_robot_commands(dribble_task.enact(True), ATTACKER_ROBOT_ID)
            elif stage == 3:
                cmd = score_goal(game, shooter_has_ball=True, shooter_id=ATTACKER_ROBOT_ID, pid_oren=pid_oren_attacker, pid_trans=pid_trans_attacker, is_yellow=ATTACKER_IS_YELLOW, shoot_in_left_goal=LEFT_GOAL)
                print(cmd)
                if cmd.kick == 1:
                    stage += 1
                robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)
        
            robot_controller_attacker.send_robot_commands()
            robot_controller_goalie.send_robot_commands()
        

def main():
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]

    game = Game()
    robot_controller_yellow = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=1
    )

    robot_controller_blue = Mock()
    
    # RealRobotController(
    #     is_team_yellow=False, game_obj=game, n_robots=1
    # )

    try:
        one_vs_one_goalie(game, robot_controller_yellow, robot_controller_yellow)
    except KeyboardInterrupt:
        print("Stopping robot.")
    finally:
        for i in range(15): # Try really hard to stop the robots!
            robot_controller_yellow.serial.write(stop_buffer_off)
            robot_controller_blue.serial.write(stop_buffer_off)

if __name__ == "__main__":
    main()
