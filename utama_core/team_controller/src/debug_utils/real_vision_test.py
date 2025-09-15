import logging
import threading
import time
from collections import deque

from utama_core.config.settings import MAX_CAMERAS, TIMESTEP
from utama_core.run.receivers import VisionReceiver

logging.basicConfig(
    filename="Utama.log",
    level=logging.CRITICAL,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def data_update_listener(receiver: VisionReceiver) -> None:
    """Continuously pull game data in the background thread."""
    receiver.pull_game_data()


def start_threads(vision_receiver: VisionReceiver) -> None:
    """Start the vision receiver in a daemon thread so it exits with the main program."""
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data, daemon=True)
    vision_thread.start()


def main():
    vision_buffers = [deque(maxlen=1) for _ in range(MAX_CAMERAS)]
    vision_receiver = VisionReceiver(vision_buffers)
    start_threads(vision_receiver)
    try:
        while True:
            start_time = time.time()
            vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffers]
            # referee_frame = ref_buffer.popleft()
            print(vision_frames)

            end_time = time.time()

            # processing_time = end_time - start_time

            # self.logger.log(
            #     logging.WARNING if processing_time > TIMESTEP else logging.INFO,
            #     "Game loop took %f secs",
            #     processing_time,
            # )

            # Sleep to maintain FPS
            wait_time = max(0, TIMESTEP - (end_time - start_time))
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
