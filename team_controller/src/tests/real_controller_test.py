from motion_planning.src.pid.pid import get_grsim_pids
from robot_control.src.skills import go_to_point
from team_controller.src.controllers import RealRobotController
from entities.game import Game
from entities.data.command import RobotCommand
import time
import queue
import logging
import threading
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data.vision_receiver import VisionDataReceiver



def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

def main():
    stop_buffer = [0, 0, 0, 0, 0, 0, 0, 0]

    pid_oren, pid_trans  = get_grsim_pids(6)

    game = Game()
    robot_controller = RealRobotController(is_team_yellow=True, game_obj=game, n_robots=1)
    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    try:
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass
            data = game.get_robot_pos(True, 1)
            if data:
                cmd = go_to_point(pid_oren, pid_trans, data, 1, (-2.25, 0), 0, False)
                print(cmd)
                robot_controller.add_robot_commands(cmd, 0)
                binary_representation = [hex(byte) for byte in robot_controller.out_packet]
                print(binary_representation)
                robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        robot_controller.serial.write(stop_buffer)
        print("Stopping main program.")


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
