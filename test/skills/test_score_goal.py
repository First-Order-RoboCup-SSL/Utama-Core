from test.common.abstract_test_manager import AbstractTestManager, TestingStatus
from team_controller.src.controllers import AbstractSimController
from config.defaults import LEFT_START_ONE, RIGHT_START_ONE
from entities.game import Game
from global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
import math
import time

from run import StrategyRunner
from strategy.skills.score_goal import ScoreGoalStrategy


class ScoreGoalTestManager(AbstractTestManager):
    """
    Test manager for the ScoreGoal strategy.
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
        ini_yellow, ini_blue = map_left_right_to_colors(
            game.my_team_is_yellow,
            game.my_team_is_right,
            RIGHT_START_ONE,
            LEFT_START_ONE,
        )

        y_robots, b_robots = map_friendly_enemy_to_colors(
            game.my_team_is_yellow, game.friendly_robots, game.enemy_robots
        )

        for i in b_robots.keys():
            sim_controller.teleport_robot(
                False, i, ini_blue[i][0], ini_blue[i][1], ini_blue[i][2]
            )
        for j in y_robots.keys():
            sim_controller.teleport_robot(
                True, j, ini_yellow[j][0], ini_yellow[j][1], ini_yellow[j][2]
            )

        # set the target robot position
        ini_pos = self.ini_pos[self.episode_i]
        sim_controller.teleport_robot(
            game.my_team_is_yellow,
            self.my_strategy.robot_id,
            ini_pos[0],
            ini_pos[1],
        )

        sim_controller.teleport_ball(0, 0)

    def eval_status(self, game: Game):
        """
        Evaluate the status of the test episode.
        """
        if game.ball.p.x > 4.5 or game.ball.p.x < -4.5:
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


def test_score_goal(
    my_team_is_yellow: bool,
    my_team_is_right: bool,
    robot_id: int,
    headless: bool,
    mode: str = "rsim",
):
    """
    Called by pytest to run the GoToBall strategy test.
    """
    runner = StrategyRunner(
        strategy=ScoreGoalStrategy(robot_id=robot_id),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=2,
    )
    test = runner.run_test(
        testManager=ScoreGoalTestManager(), episode_timeout=10, rsim_headless=headless
    )
    assert test


if __name__ == "__main__":
    # This is just for running the test manually, not needed for pytest
    test_score_goal(
        my_team_is_yellow=True,
        my_team_is_right=False,
        robot_id=0,
        mode="rsim",
        headless=False,
    )
