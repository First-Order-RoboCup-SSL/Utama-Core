import time
from motion_planning.src.pid.pid import get_real_pids
from team_controller.src.controllers import RealRobotController
from entities.game import Game
from global_utils.math_utils import distance
from robot_control.src.intent import score_goal, PassBall
import queue
import threading
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data.vision_receiver import VisionDataReceiver
from robot_control.src.skills import go_to_ball, turn_on_spot, go_to_point
import logging

logger = logging.getLogger(__name__)

MAX_TIME = 20  # in seconds
N_YELLOW = 2
N_BLUE = 5

r0 = 0
r1 = 1

receiving_point = (-3, 2)
passing_point = (-1.5, -2)


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def read_to_game(message_queue: queue.SimpleQueue, game: Game):
    (message_type, message) = message_queue.get()  # Infinite timeout for now

    if message_type == MessageType.VISION:
        game.add_new_state(message)
        vision_recieved = True
    elif message_type == MessageType.REF:
        pass


def test_rsim_2v5():
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""
    # pid_oren, pid_trans = get_
    game = Game()

    # Shooting team gets full complement of robots, defending team only half

    pid_oren, pid_trans = get_real_pids(N_YELLOW)
    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, n_cameras=1)
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    robot_controller = RealRobotController(
        is_team_yellow=True, game_obj=game, n_robots=2
    )
    target_ball_pos = (game.ball.x, game.ball.y)

    try:
        while distance((game.ball.x, game.ball.y), target_ball_pos) < 0.1:
            read_to_game(message_queue, game)
            print("Getting to ball")
            r1_data = game.get_robot_pos(True, r1)
            cmd1 = go_to_ball(
                pid_oren,
                pid_trans,
                r1_data,
                r1,
                game.ball,
            )
            robot_controller.add_robot_commands(cmd1, r1)
            robot_controller.send_robot_commands()
        # r1 get to pos
        while True:
            read_to_game(message_queue, game)
            print("Getting to pos")
            r1_data = game.get_robot_pos(True, r1)
            r0_data = game.get_robot_pos(True, r0)
            if distance((r0_data.x, r0_data.y), receiving_point) < 0.05:
                break
            cmd0 = go_to_point(
                pid_oren,
                pid_trans,
                r0_data,
                r0,
                receiving_point,
                r0_data.orientation,
            )
            cmd1 = go_to_point(
                pid_oren,
                pid_trans,
                r1_data,
                r1,
                passing_point,
                r1_data.orientation,
                True,
            )

            robot_controller.add_robot_commands(cmd1, r1)
            robot_controller.add_robot_commands(cmd0, r0)
            robot_controller.send_robot_commands()

        # r1 pass to r0
        pass_task1 = PassBall(pid_oren, pid_trans, game, r1, r0, receiving_point)
        start_t = time.time()
        while time.time() - start_t < 5:
            read_to_game(message_queue, game)
            print("Passing")
            cmd1, cmd0 = pass_task1.enact(True)
            robot_controller.add_robot_commands(cmd0, r0)
            robot_controller.add_robot_commands(cmd1, r1)
            robot_controller.send_robot_commands()

        del pass_task1

        # r0 score goal
        while not game.is_ball_in_goal(our_side=False):
            read_to_game(message_queue, game)
            print("scoring goal")
            cmd0 = score_goal(game, True, r0, pid_oren, pid_trans, True, True)
            robot_controller.add_robot_commands(cmd0, r0)
            robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")
