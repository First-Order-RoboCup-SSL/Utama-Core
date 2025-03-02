import time
import threading
import logging
import warnings

import py_trees

from config.settings import MAX_CAMERAS, TIMESTEP
from entities.game import Game
from collections import deque
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import get_grsim_pids
from receivers.referee_receiver import RefereeMessageReceiver
from refiners.has_ball import HasBallRefiner
from refiners.position import PositionRefiner
# from refiners.referee import RefereeRefiner
from refiners.velocity import VelocityRefiner
from receivers.vision_receiver import VisionReceiver
from collections.abc import Callable
from run import GameGater
# from strategy.startup_strategy import StartupStrategy
from strategy.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.strategy import Strategy
from team_controller.src.controllers.common.robot_controller_abstract import AbstractRobotController
from team_controller.src.controllers.sim.grsim_robot_controller import GRSimRobotController

def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

def start_threads(vision_receiver, referee_receiver):
    # Start the data receiving in separate threads
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)

    # Allows the thread to close when the main program exits
    vision_thread.daemon = True
    referee_thread.daemon = True

    # Start both thread
    vision_thread.start()
    referee_thread.start()

def main(strategy: Strategy, robot_controller: AbstractRobotController):
    logger = logging.getLogger(__name__)
    warnings.simplefilter("default", DeprecationWarning)
    
    game = Game(my_team_is_yellow=True, num_friendly_robots=6, num_enemy_robots=6)

    robot_buffer = deque(maxlen=1) # TODO: Add separate thread to read robot data when we have it
    vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
    ref_buffer = deque(maxlen=1)

    referee_receiver = RefereeMessageReceiver(ref_buffer, debug=False)
    vision_receiver = VisionReceiver(vision_buffers)
    
    start_threads(vision_receiver, referee_receiver)

    game = GameGater().wait_until_game_valid(True, True, 6,6,True,vision_buffers, position_refiner)
    
    position_refiner = PositionRefiner()
    velocity_refiner = VelocityRefiner()
    # hasball_refiner = HasBallRefiner()
    # referee_refiner = RefereeRefiner()


    
    present_future_game = PresentFutureGame(game) 

    while True:
        start_time = time.time()
        vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffers]
        robot_frame = robot_buffer.popleft()
        referee_frame = ref_buffer.popleft()

        game = position_refiner.refine(game, vision_frames)
        game = velocity_refiner.refine(game, robot_frame.imu_data)    
        # game = hasball_refiner.refine(game, robot_frame.ir_data)
        # game = referee_refiner.refine(game, referee_frame)

        present_future_game.add_new_game(game)
        
        commands = strategy.step(present_future_game)
        for robot_id, command in commands.items():
            robot_controller.add_robot_commands(command, robot_id)
            
        end_time = time.time()
        time.sleep(TIMESTEP - (end_time - start_time))

if __name__ == "__main__":
    sim_robot_controller = GRSimRobotController(is_team_yellow=True)
    # startup_strategy = StartupStrategy(*get_grsim_pids())
    bt = py_trees.behaviours.SuccessEveryN("EveryN", 5)
    main(BehaviourTreeStrategy(sim_robot_controller, bt), sim_robot_controller)
