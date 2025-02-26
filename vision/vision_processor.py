from entities.data.vision import FrameData
from entities.game import Game

# Move averaging into here
class VisionProcessor:
    def __init__(self, n_expected_robots_yellow: int, n_expected_robots_blue: int, n_expected_balls: int):
        self.n_expected_robots_yellow = n_expected_robots_yellow
        self.n_expected_robots_blue = n_expected_robots_blue
        self.n_expected_balls = n_expected_balls
        self.frame = FrameData(0, [], [], [])

    def is_ready(self) -> bool:
        return (len(self.frame.blue_robots) == self.n_expected_robots_blue
               and len(self.frame.yellow_robots) == self.n_expected_robots_yellow 
               and len(self.frame.n_expected_balls) == self.n_expected_balls)

    def add_new_state(self, frame_data: FrameData):
        

    def get_game(self):
        return self.frame
