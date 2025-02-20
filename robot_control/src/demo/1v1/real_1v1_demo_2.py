from calendar import c
from entities.data.command import RobotCommand
from motion_planning.src.pid.pid import get_real_pids, get_real_pids_goalie
from robot_control.src.intent import score_goal
from robot_control.src.skills import go_to_point, go_to_ball, goalkeep, turn_on_spot
from team_controller.src.controllers import RealRobotController
from entities.game import Game
import queue
import logging
import threading
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data.vision_receiver import VisionDataReceiver
import numpy as np
import random
import logging
from mock import Mock
import math
import time

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

    stage2_target_x = -3
    stage2_target_y = 0.5

    stage3_target_x = -3
    stage3_target_y = 1.5

    stage = 1
    start = None
    iter = 0
    ball_init_pos = None
    turnedits = 0
    while not goal_scored:
        iter += 1
        (message_type, message) = message_queue.get()
        if message_type == MessageType.VISION:
            game.add_new_state(message)
            print(message)
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
            
            # goalie_command = goalkeep(True, game, GOALIE_ROBOT_ID, pid_oren_goalie, pid_trans_goalie, not ATTACKER_IS_YELLOW, False)
            # robot_controller_goalie.add_robot_commands(goalie_command, GOALIE_ROBOT_ID)
            # print("GOALIE COMMAND", goalie_command)
            print(f"STAGE {stage}")

            if stage == 1:
                if robot_controller_attacker.robot_has_ball(ATTACKER_ROBOT_ID):
                    stage += 1
                else:
                    robot_controller_attacker.add_robot_commands(go_to_ball(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), ATTACKER_ROBOT_ID, game.ball), ATTACKER_ROBOT_ID)    
            elif stage == 2:                
                if math.dist((attacker_pos.x, attacker_pos.y), (stage2_target_x, stage2_target_y)) < 0.07 and stage == 2:
                    stage += 1
                else:
                    cmd = go_to_point(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), 1, (stage2_target_x, stage2_target_y), None, dribbling=True)
                    print("GO TO POINT", cmd)
                    robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)
            elif stage == 3:
                target_orientation = math.pi / 2
                if abs(target_orientation - attacker_pos.orientation) < 0.05:
                    stage += 1
                else:
                    cmd = turn_on_spot(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), 1, target_orientation, dribbling=False, pivot_on_ball=True)
                    robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)                
            elif stage == 4:
                print("STAGE 4")
                if not start:
                    start = time.time()
                if time.time() - start > 0.1:
                    stage += 1
                else:
                    cmd = RobotCommand(
                        local_left_vel=-0.2,
                        local_forward_vel=0,
                        angular_vel=0,
                        kick=0,
                        chip=0,
                        dribble=0
                    ) 
                    robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)
            elif stage == 5:
                print("DIST", math.dist((attacker_pos.x, attacker_pos.y), (stage3_target_x, stage3_target_y)))
                if math.dist((attacker_pos.x, attacker_pos.y), (stage3_target_x, stage3_target_y)) < 0.07:
                    stage += 1
                else:
                    cmd = go_to_point(pid_oren_attacker, pid_trans_attacker, game.get_robot_pos(ATTACKER_IS_YELLOW, ATTACKER_ROBOT_ID), 1, (stage3_target_x, stage3_target_y), None, dribbling=True)
                    print("GO TO POINT", cmd)
                    robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)
            elif stage == 6:
                cmd = score_goal(game, shooter_has_ball=True, shooter_id=ATTACKER_ROBOT_ID, pid_oren=pid_oren_attacker, pid_trans=pid_trans_attacker, is_yellow=ATTACKER_IS_YELLOW, shoot_in_left_goal=LEFT_GOAL)
                print("SENDING COMMAND", cmd)
                if cmd.kick == 1:
                    stage += 1
                    
                robot_controller_attacker.add_robot_commands(cmd, ATTACKER_ROBOT_ID)
            robot_controller_attacker.send_robot_commands()
            # robot_controller_goalie.send_robot_commands()
        
        
def main():
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]

    game = Game()
    robot_controller_yellow = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=2
    )

    # robot_controller_yellow = Mock()
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
