from calendar import c
from motion_planning.src.pid.pid import get_real_pids
from robot_control.src.skills import (
    go_to_point,
    go_to_ball,
    turn_on_spot,
    empty_command,
)
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


def rotate_on_ball_with_vision(game: Game, robot_controller: RealRobotController):
    pid_oren, pid_trans = get_real_pids(6)
    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    initial_ball_pos = None
    go_back = 10000
    done = False
    vision_recieved = False
    oren_done = False
    while True:
        (message_type, message) = message_queue.get()  # Infinite timeout for now

        if message_type == MessageType.VISION:
            game.add_new_state(message)
            vision_recieved = True
        elif message_type == MessageType.REF:
            pass

        data = game.get_robot_pos(True, 1)
        if vision_recieved:
            if not initial_ball_pos:
                initial_ball_pos = game.ball

            my_pos = game.get_robot_pos(True, 1)
            if my_pos is not None:
                distance = np.hypot(
                    my_pos.x - initial_ball_pos.x, my_pos.y - initial_ball_pos.y
                )

                SLOW_FRAMES = 45
                print("DIST", distance)
                if abs(distance) < 0.2 and not done:
                    print("SWITCHING TO TURN")
                    go_back = 2 * SLOW_FRAMES
                    done = True
                elif go_back <= SLOW_FRAMES:
                    print("TURNING NOW")
                    cmd = turn_on_spot(
                        pid_oren, pid_trans, data, 1, np.pi / 2, True, True
                    )
                    print(cmd)
                elif go_back <= 2 * SLOW_FRAMES:
                    print("SLOWING")
                    cmd = RobotCommand(
                        (go_back - SLOW_FRAMES) / (SLOW_FRAMES),
                        0,
                        0,
                        False,
                        False,
                        True,
                    )
                    go_back -= 1
                else:
                    cmd = go_to_ball(pid_oren, pid_trans, data, 1, game.ball)

                if oren_done:
                    cmd = RobotCommand(0.5, 0, 0, False, False, True)

                robot_controller.add_robot_commands(cmd, 0)
                robot_controller.send_robot_commands()

        vision_recieved = False


def get_ball_test_with_vision(game: Game, robot_controller: RealRobotController):
    pid_oren, pid_trans = get_real_pids(6)
    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    initial_ball_pos = None
    go_back = 10000
    done = False
    vision_recieved = False
    oren_done = False
    while True:
        (message_type, message) = message_queue.get()  # Infinite timeout for now

        if message_type == MessageType.VISION:
            game.add_new_state(message)
            vision_recieved = True
        elif message_type == MessageType.REF:
            pass

        data = game.get_robot_pos(True, 1)
        if vision_recieved:
            if not initial_ball_pos:
                initial_ball_pos = game.ball

            my_pos = game.get_robot_pos(True, 1)
            if my_pos is not None:
                distance = np.hypot(
                    my_pos.x - initial_ball_pos.x, my_pos.y - initial_ball_pos.y
                )

                SLOW_FRAMES = 200
                print("DIST", distance)
                if abs(distance) < 0.2 and not done:
                    print("SWITCHING TO BACK")
                    go_back = 2 * SLOW_FRAMES
                    done = True

                if go_back == 0:
                    cmd = RobotCommand(-0.5, 0, 0, False, False, True)
                    print(cmd)
                    print("GOING BACK NOW")
                    oren_done = True
                elif go_back <= SLOW_FRAMES:
                    print("SLOWING BACK")
                    cmd = RobotCommand(
                        -(SLOW_FRAMES - go_back) / (SLOW_FRAMES),
                        0,
                        0,
                        False,
                        False,
                        True,
                    )
                    print(cmd)
                    go_back -= 1
                elif go_back <= 2 * SLOW_FRAMES:
                    print("SLOWING")
                    cmd = RobotCommand(
                        (go_back - SLOW_FRAMES) / (SLOW_FRAMES),
                        0,
                        0,
                        False,
                        False,
                        True,
                    )
                    go_back -= 1
                else:
                    cmd = go_to_ball(pid_oren, pid_trans, data, 1, game.ball)

                if oren_done:
                    cmd = RobotCommand(0.5, 0, 0, False, False, True)

                # cmd = go_to_point(pid_oren, pid_trans, data, 1, (-2, -0.5), 0, False)

                robot_controller.add_robot_commands(cmd, 1)
                robot_controller.send_robot_commands()

        vision_recieved = False


def test_command(
    robot_controller: RealRobotController,
    robot_id: int,
    ramp_iters: int,
    ramp_only: bool = False,
    dribble: bool = False,
):
    iter = 0
    stop_its = 100
    start_time = time.time()
    while True:
        if ramp_only and iter > ramp_iters:
            break
        iter += 1
        print(iter)
        if stop_its == iter:
            dribble = not dribble
        cmd = RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            # angular_vel=min(1, iter / ramp_iters) * target_val,
            kick=0,
            chip=0,
            dribble=dribble,
        )

        robot_controller.add_robot_commands(cmd, robot_id)
        # binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
        # print(
        #     f"command sent!\n",
        # )
        # print(binary_representation)
        robot_controller.send_robot_commands()
        print(robot_controller.robot_has_ball(robot_id))
        start_time = time.time()
        time.sleep(0.017)


def test_kicker(robot_controller: RealRobotController, robot_id: int, dribbler_on=True):
    cmd0 = RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=1,
        chip=0,
        dribble=0,
    )
    # for _ in range(15):
    #     robot_controller.add_robot_commands(
    #         empty_command(dribbler_on=dribbler_on), robot_id
    #     )
    #     robot_controller.send_robot_commands()
    #     time.sleep(0.05)
    robot_controller.add_robot_commands(cmd0, robot_id)
    robot_controller.send_robot_commands()
    time.sleep(0.05)
    robot_controller.add_robot_commands(empty_command(), robot_id)
    robot_controller.send_robot_commands()
    time.sleep(0.05)


def main():
    robot_id = 1
    stop_buffer_off = [0, 0, 0, 0, 0, 0, 0, 0]

    game = Game()
    robot_controller = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=2
    )
    try:
        # test_command(robot_controller, robot_id, 100, False, True)
        # get_ball_test_with_vision(game, robot_controller)
        test_kicker(robot_controller, 1, dribbler_on=True)
    finally:
        # try to stop the robot 15 times
        print("Stopping robot.")

        for _ in range(15):
            robot_controller.send_robot_commands()
        robot_controller.serial.close()

    # binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    # print(binary_representation)
    # robot_controller.send_robot_commands()


if __name__ == "__main__":
    main()
