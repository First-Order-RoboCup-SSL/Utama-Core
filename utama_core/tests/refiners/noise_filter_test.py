# import math

# from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
# from utama_core.entities.game import Game
# from utama_core.global_utils.mapping_utils import (
#     map_friendly_enemy_to_colors,
#     map_left_right_to_colors,
# )
# from utama_core.run import StrategyRunner
# from utama_core.strategy import DefenceStrategy, RobotPlacementStrategy, StartupStrategy
# from utama_core.team_controller.src.controllers import AbstractSimController
# from utama_core.tests.common.abstract_test_manager import (
#     AbstractTestManager,
#     TestingStatus,
# )


# class GoToBallTestManager(AbstractTestManager):
#     """Test manager for the GoToBall strategy."""

#     def __init__(self):
#         super().__init__()
#         self.n_episodes = 1
#         self.ini_pos = self._generate_ini_pos()

#     def reset_field(self, sim_controller: AbstractSimController, game: Game):
#         """Reset position of robots and ball for the next strategy test."""
#         # Reset all other robots
#         ini_yellow, ini_blue = map_left_right_to_colors(
#             game.my_team_is_yellow,
#             game.my_team_is_right,
#             RIGHT_START_ONE,
#             LEFT_START_ONE,
#         )

#         y_robots, b_robots = map_friendly_enemy_to_colors(
#             game.my_team_is_yellow, game.friendly_robots, game.enemy_robots
#         )

#         for i in b_robots.keys():
#             sim_controller.teleport_robot(False, i, ini_blue[i][0], ini_blue[i][1], ini_blue[i][2])
#         for j in y_robots.keys():
#             sim_controller.teleport_robot(True, j, ini_yellow[j][0], ini_yellow[j][1], ini_yellow[j][2])

#         # set the target robot position
#         # ini_pos = self.ini_pos[self.episode_i]
#         # sim_controller.teleport_robot(
#         #     game.my_team_is_yellow,
#         #     self.my_strategy.robot_id,
#         #     ini_pos[0],
#         #     ini_pos[1],
#         # )

#         sim_controller.teleport_ball(0, 0)

#     def eval_status(self, game: Game):
#         """Evaluate the status of the test episode."""
#         # TODO
#         return TestingStatus.SUCCESS
#         #return TestingStatus.IN_PROGRESS

#     def get_n_episodes(self):
#         """Get the number of episodes to run for the test."""
#         return len(self.ini_pos)

#     def _generate_ini_pos(self, radius=2):
#         positions = []
#         for i in range(self.n_episodes):
#             angle = 2 * math.pi * i / self.n_episodes  # angle in radians
#             x = radius * math.cos(angle)
#             y = radius * math.sin(angle)
#             positions.append((x, y))
#         return positions


# def test_go_to_ball(
#     my_team_is_yellow: bool,
#     my_team_is_right: bool,
#     headless: bool,
#     mode: str = "grsim",
# ):
#     """Called by pytest to run the GoToBall strategy test."""
#     runner = StrategyRunner(
#         strategy=StartupStrategy(),
#         my_team_is_yellow=my_team_is_yellow,
#         my_team_is_right=my_team_is_right,
#         mode=mode,
#         exp_friendly=6,
#         exp_enemy=0,
#     )
#     test = runner.run_test(testManager=GoToBallTestManager(), episode_timeout=10, rsim_headless=headless)
#     assert test


# if __name__ == "__main__":
#     # This is just for running the test manually, not needed for pytest
#     test_go_to_ball(
#         my_team_is_yellow=True,
#         my_team_is_right=True,
#         robot_id=0,
#         mode="grsim",
#         headless=False,
#     )
