import time
import threading
import logging

from config.settings import TIMESTEP
from entities.game import Game
from collections import deque
from refiners.has_ball import HasBallRefiner
from refiners.position import PositionRefiner
from refiners.referee import RefereeRefiner
from refiners.velocity import VelocityRefiner
from team_controller.src.data import RefereeMessageReceiver
from vision.vision_receiver import VisionReceiver

logger = logging.getLogger(__name__)

import warnings

# Enable all warnings, including DeprecationWarning
warnings.simplefilter("default", DeprecationWarning)


def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


class GameGater:
    
    def __init__(self, robot_buffer, vision_buffers, ref_buffer):
        self.robot_buffer = robot_buffer
        self.vision_buffer = vision_buffers
        self.ref_buffer = ref_buffer
        self.yrs = set()
        self.brs = set()
        self.balls = False

    def not_ready(self, exp_yellow: int, exp_blue: int, exp_ball: bool):
        for vb in self.vision_buffers:
            if vb:
                ff = vb.popleft()
                self.yrs = self.yrs.union({r.robot_id for r in ff.yellow_robots})
                self.brs = self.brs.union({r.robot_id for r in ff.blue_robots})
                self.balls = self.balls or bool(ff.balls)
        
        return len(self.yrs) < exp_yellow or len(self.brs) < exp_blue or self.balls != exp_ball


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

MAX_CAMERAS = 10
def main():
    game = Game(my_team_is_yellow=True, num_friendly_robots=6, num_enemy_robots=6)

    robot_buffer = deque(maxlen=1) # TODO: Add separate thread to read robot data when we have it
    vision_buffer = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
    ref_buffer = deque(maxlen=1)

    referee_receiver = RefereeMessageReceiver(ref_buffer, debug=False)
    vision_receiver = VisionReceiver(vision_buffer)
    
    start_threads(vision_receiver, referee_receiver)

    gamegater = GameGater(robot_buffer, vision_buffer, ref_buffer)
    
    position_refiner = PositionRefiner()
    velocity_refiner = VelocityRefiner()
    hasball_refiner = HasBallRefiner()
    referee_refiner = RefereeRefiner()

    while gamegater.not_ready():
        vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffer]
        game = position_refiner.refine(game, vision_frames)
    
    present_future_game = PresentFutureGame(game, Predictor1, Predictor2...) 
    strategy = StartUpStrategy(present_future_game)

    while not strategy.done():
        start_time = time.time()
        vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffer]
        robot_frame = robot_buffer.popleft()
        referee_frame = ref_buffer.popleft()

        game = position_refiner.refine(game, vision_frames)
        game = velocity_refiner.refine(game, robot_frame.imu_data)    
        game = hasball_refiner.refine(game, robot_frame.ir_data)
        game = referee_refiner.refine(game, referee_frame)

        present_future_game.add_new_game(game)
        strategy.enact()
        end_time = time.time()
        time.sleep(TIMESTEP - (end_time - start_time))

if __name__ == "__main__":
    main()
