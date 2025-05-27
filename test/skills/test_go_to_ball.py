from test.common.abstract_test_manager import AbstractTestManager, TestingStatus
from team_controller.src.controllers import AbstractSimController
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.game.game import Game
import math
import time

from run import StrategyRunner
from strategy.skills.go_to_ball import GoToBallStrategy


class GoToBallTestManager(AbstractTestManager):
    """
    Test manager for the GoToBall strategy.
    """

    def __init__(self):
        super().__init__()
        self.n_episodes = 3
        self.ini_pos = self._generate_ini_pos()

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """
        Reset position of robots and ball for the next strategy test.
        """
        # Reset all other robots
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

        sim_controller.teleport_ball(0, 0)

    def eval_status(self, game: Game):
        """
        Evaluate the status of the test episode.
        """
        if game.friendly_robots[self.my_strategy.target_id].has_ball:
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS

    def get_n_episodes(self):
        """
        Get the number of episodes to run for the test.
        """
        return len(self.ini_pos)

    def _generate_ini_pos(self, radius=2):
        positions = []
        for i in range(self.n_episodes):
            angle = 2 * math.pi * i / self.n_episodes  # angle in radians
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append((x, y))
        return positions


def test_go_to_ball(
    my_team_is_yellow: bool,
    my_team_is_right: bool,
    target_id: int,
    headless: bool,
    mode: str = "rsim",
):
    """
    Called by pytest to run the GoToBall strategy test.
    """
    runner = StrategyRunner(
        strategy=GoToBallStrategy(target_id=target_id),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=3,
        exp_enemy=3,
        exp_ball=True,
    )
    test = runner.run_test(
        testManager=GoToBallTestManager(), episode_timeout=10, rsim_headless=headless
    )
    assert test


if __name__ == "__main__":
    # This is just for running the test manually, not needed for pytest
    test_go_to_ball(
        my_team_is_yellow=True,
        my_team_is_right=True,
        target_id=0,
        mode="rsim",
        headless=False,
    )
