import logging
import threading
import time
from collections import deque

from run.receivers.vision_receiver import VisionReceiver

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def start_threads(vision_receiver: VisionReceiver):
    # Start the data receiving in separate threads
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)

    # Allows the thread to close when the main program exits
    vision_thread.daemon = True

    # Start both thread
    vision_thread.start()


def main():
    # Runs the vision receiver,
    vision_buffer = [deque(maxlen=1) for _ in range(4)]
    vision_receiver = VisionReceiver(vision_buffer)
    start_threads(vision_receiver)

    prog_start = time.time()

    for _ in range(100000):
        for cid, vb in enumerate(vision_buffer):
            if vb:
                print(f"Camera {cid} has data: {vb[0]} at {time.time() - prog_start}")
            else:
                print(f"Camera {cid} has no data at {time.time() - prog_start}")
        time.sleep(1)


if __name__ == "__main__":
    main()
