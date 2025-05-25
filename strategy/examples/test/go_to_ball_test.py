from run import AbstractTestManager, TestStatus
from team_controller.src.controllers import AbstractSimController
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.game.game import Game
import math
import time


class GoToBallTestManager(AbstractTestManager):
    """
    Test manager for the GoToBall strategy.
    """

    def __init__(self):
        super().__init__()
        self.n_episodes = 10
        self.ini_pos = self.generate_ini_pos()

    def generate_ini_pos(self, radius=2):
        positions = []
        for i in range(self.n_episodes):
            angle = 2 * math.pi * i / self.n_episodes  # angle in radians
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """
        Reset position of robots and ball for the next strategy test.
        """
        # Reset all other robots
        sim_controller.teleport_ball(0, 0)
        yellow_is_right = game.my_team_is_yellow == game.my_team_is_right
        if yellow_is_right:
            ini_yellow = RIGHT_START_ONE
            ini_blue = LEFT_START_ONE
        else:
            ini_yellow = LEFT_START_ONE
            ini_blue = RIGHT_START_ONE

        if game.my_team_is_yellow:
            self.y_robots = game.friendly_robots
            self.b_robots = game.enemy_robots
        else:
            self.y_robots = game.enemy_robots
            self.b_robots = game.friendly_robots
        print(self.y_robots)
        for i in self.b_robots.keys():
            sim_controller.teleport_robot(
                False, i, ini_blue[i][0], ini_blue[i][1], ini_blue[i][2]
            )
        for j in self.y_robots.keys():
            sim_controller.teleport_robot(
                True, j, ini_yellow[j][0], ini_yellow[j][1], ini_yellow[j][2]
            )

        # set the target robot position
        ini_pos = self.ini_pos[self.ep_n]
        sim_controller.teleport_robot(
            game.my_team_is_yellow,
            self.my_strategy.target_id,
            ini_pos[0],
            ini_pos[1],
        )

    def eval_status(self, game: Game):
        """
        Evaluate the status of the test episode.
        """
        THRESHOLD = 0.2
        rb_p = game.friendly_robots[self.my_strategy.target_id].p
        ball_p = game.ball.p
        if math.dist((rb_p.x, rb_p.y), (ball_p.x, ball_p.y)) < THRESHOLD:
            return TestStatus.SUCCESS
        return TestStatus.IN_PROGRESS

    def get_n_episodes(self):
        """
        Get the number of episodes to run for the test.
        """
        return len(self.ini_pos)
