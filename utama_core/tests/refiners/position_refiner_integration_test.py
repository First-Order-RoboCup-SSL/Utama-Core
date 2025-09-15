import logging
import threading
import time
from collections import deque

from utama_core.run import GameGater
from utama_core.run.receivers.vision_receiver import VisionReceiver
from utama_core.run.refiners import PositionRefiner

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def start_threads(vision_receiver):
    # Start the data receiving in separate threads
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)

    # Allows the thread to close when the main program exits
    vision_thread.daemon = True

    # Start both thread
    vision_thread.start()


def main():
    # Runs the vision receiver,
    vision_buffers = [deque(maxlen=1) for _ in range(4)]
    vision_receiver = VisionReceiver(vision_buffers)
    position_refiner = PositionRefiner()

    start_threads(vision_receiver)
    NUM_FRIENDLY = 1
    NUM_ENEMY = 1

    print("Waiting for game to be valid...")
    game = GameGater.wait_until_game_valid(True, True, 1, 1, True, vision_buffers, position_refiner, False)

    prog_start = time.time()

    for _ in range(600):
        frames = []
        for cid, vb in enumerate(vision_buffers):
            if vb:
                print(f"Camera {cid} has data: {vb[0]} at {time.time() - prog_start}")
                frames.append(vb.popleft())
            else:
                print(f"Camera {cid} has no data at {time.time() - prog_start}")
        game = position_refiner.refine(game, frames)
        print(game)
        assert len(game.friendly_robots) == NUM_FRIENDLY
        assert len(game.enemy_robots) == NUM_ENEMY
        time.sleep(0.0167)


if __name__ == "__main__":
    main()
