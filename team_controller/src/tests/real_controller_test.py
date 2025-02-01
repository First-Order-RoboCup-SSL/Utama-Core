from calendar import c
from motion_planning.src.pid.pid import get_real_pids
from robot_control.src.skills import go_to_point, go_to_ball
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

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def test_with_vision(game: Game, robot_controller: RealRobotController):
    pid_oren, pid_trans = get_real_pids(6)
    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    initial_ball_pos = None
    go_back = 100000000000000000000
    done = False
    while True:
        (message_type, message) = message_queue.get()  # Infinite timeout for now
        if message_type == MessageType.VISION:
            game.add_new_state(message)
        elif message_type == MessageType.REF:
            pass
        data = game.get_robot_pos(True, 1)
        if data:
            if not initial_ball_pos:
                initial_ball_pos = game.get_ball_pos()[0]
            
            my_pos = game.get_robot_pos(True, 1)
            distance = np.hypot(my_pos.x - initial_ball_pos.x, my_pos.y - initial_ball_pos.y)

            SLOW_FRAMES = 50
            print("DIST", distance)
            if distance < 0.2 and not done:
                print("SWITCHING TO BACK")
                go_back = 2 * SLOW_FRAMES
                done = True

            if go_back == 0:
                cmd = RobotCommand(-0.5, 0, 0, False, False, True)
                print("GOING BACK NOW")
            elif go_back <= SLOW_FRAMES:
                print("SLOWING BACK")
                cmd = RobotCommand(-(SLOW_FRAMES - go_back)/(SLOW_FRAMES), 0, 0, False, False, True)
                go_back -= 1    
            elif go_back <= 2 * SLOW_FRAMES:
                print("SLOWING")
                cmd = RobotCommand((go_back - SLOW_FRAMES) / (SLOW_FRAMES), 0, 0, False, False, True)
                go_back -= 1
            else:
                cmd = go_to_ball(pid_oren, pid_trans, data, 1, game.ball)
            # cmd = go_to_point(pid_oren, pid_trans, data, 1, (-2, -0.5), 0, False)
                            
            robot_controller.add_robot_commands(cmd, 0)
            robot_controller.send_robot_commands()

def test_rotation(robot_controller: RealRobotController, target_val: int, ramp_iters: int, ramp_only: bool=False, dribble: bool = False):
    iter = 0
    start_time = time.time()
    while True:
        if ramp_only and iter > ramp_iters:
            break
        iter += 1 
        cmd = RobotCommand(
            local_forward_vel=0, 
            local_left_vel=0, 
            angular_vel=min(1, iter / ramp_iters) * target_val, 
            kick=0, 
            chip= 0, 
            dribble=dribble,
        )
    
        robot_controller.add_robot_commands(cmd, 0)
        binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
        print(f"command sent!\n",)
        print(binary_representation)
        robot_controller.send_robot_commands()
        start_time = time.time()
        time.sleep(0.017)

def main():
    stop_buffer_on = [0, 0, 0, 0, 0, 0, 32, 0]
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]

    game = Game()
    robot_controller = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=1
    )
    try:
        test_rotation(robot_controller, 0, 100, False, True)
        # test_with_vision(game, robot_controller)
    except KeyboardInterrupt:
        # try to stop the robot 15 times
        print("Stopping robot.")
        test_rotation(robot_controller, 0, 100, True, True)

        for i in range(15):
            robot_controller.serial.write(stop_buffer_on if i % 2 else stop_buffer_off)
        robot_controller.serial.close()

    # binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    # print(binary_representation)
    # robot_controller.send_robot_commands()

    # Start the data receiving in a separate thread

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()


if __name__ == "__main__":
    main()
