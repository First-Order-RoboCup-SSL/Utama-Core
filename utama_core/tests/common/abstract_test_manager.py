from abc import ABC, abstractmethod
from enum import Enum

from utama_core.entities.game import Game
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import AbstractSimController


class TestingStatus(Enum):
    """Enum to represent the status of a test episode."""

    __test__ = False  # To prevent pytest from trying to collect this as a test case

    SUCCESS = 0
    FAILURE = 1
    IN_PROGRESS = 2


class AbstractTestManager(ABC):
    def __init__(self):
        self.current_episode_number = 0
        self.my_strategy: AbstractStrategy = None
        self.opp_strategy: AbstractStrategy = None

    def load_strategies(self, my_strategy: AbstractStrategy, opp_strategy: AbstractStrategy):
        """Method is called at start of strategyRunner.run_test Load the strategy to be tested."""
        self.my_strategy = my_strategy
        self.opp_strategy = opp_strategy

    def update_episode_n(self, current_episode_number: int):
        """Method is used to sync test_manager on the iteration number that strategyRunner thinks it is on."""
        self.current_episode_number = current_episode_number

    ### START OF FUNCTIONS TO BE IMPLEMENTED FOR YOUR MANAGER ###

    @abstractmethod
    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """
        Method is called at start of each test episode in strategyRunner.run_test().
        Use this to reset position of robots and ball for the next episode.
        Args:
            sim_controller (AbstractSimController): The simulation controller to manipulate robot positions.
            game (Game): The current game state.
        """
        ...

    @abstractmethod
    def eval_status(self, game: Game) -> TestingStatus:
        """
        Method is called on each iteration in strategyRunner.run_test Evaluate the status of the test episode.

        Returns the current status of the test episode:
        - TestingStatus.SUCCESS: test passed (terminate the episode with success)
        - TestingStatus.FAILURE: test failed (terminate the episode with failure)
        - TestingStatus.IN_PROGRESS: test still ongoing (continue running the episode)
        """
        ...

    @abstractmethod
    @property
    def n_episodes(self) -> int:
        """
        Specify here the number of episodes the test manager should run.

        Returns:
            int: Number of episodes to run.
        """
        ...

    ### END OF FUNCTIONS TO BE IMPLEMENTED FOR YOUR MANAGER ###
