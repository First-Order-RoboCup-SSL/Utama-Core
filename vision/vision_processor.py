from config.settings import TIMESTEP
from entities.data.vision import FrameData

# Move averaging into here
# and fixing out of order
class VisionProcessor:
    """Puts processed vision data into queue:
        - 60 fps
        - No empty values (all extrapolated)
        Only starts when we have enough data
    """
    def __init__(self, n_expected_robots_yellow: int, n_expected_robots_blue: int, n_expected_balls: int, queue: SimpleQueue):
        self.n_expected_robots_yellow = n_expected_robots_yellow
        self.n_expected_robots_blue = n_expected_robots_blue
        self.n_expected_balls = n_expected_balls
        self.frame = FrameData(0, [], [], [])
        self.queue = queue
        self.last_enqueue = 0
        self.highest_seen_packet = None

    def is_ready(self) -> bool:
        # return (len(self.frame.blue_robots) == self.n_expected_robots_blue
        #        and len(self.frame.yellow_robots) == self.n_expected_robots_yellow 
        #        and len(self.frame.n_expected_balls) == self.n_expected_balls)

    def add_new_frame(self, frame_data: DirtyFrameData):
        if self.highest_seen_packet.ts - self.last_enqueue > TIMESTEP and self.is_ready():
            queue.push() # Pself.extrapolate(self.highest_seen_packet))
            self.last_enqueue = self.highest_seen_packet

        if frame_data.ts > self.highest_seen_packet.ts:
            self.highest_seen_packet = frame_data
