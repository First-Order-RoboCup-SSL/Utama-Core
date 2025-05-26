from team_controller.src.controllers import AbstractSimController
from strategy.abstract_strategy import AbstractStrategy
from entities.game import Game
from enum import Enum
from abc import ABC, abstractmethod


class TestStatus(Enum):
    """
    Enum to represent the status of a test episode.
    """

    SUCCESS = 0
    FAILURE = 1
    IN_PROGRESS = 2


class AbstractTestManager(ABC):
    def __init__(self):
        self.ep_n = 0
        self.my_strategy: AbstractStrategy = None
        self.opp_strategy: AbstractStrategy = None

    def load_strategies(
        self, my_strategy: AbstractStrategy, opp_strategy: AbstractStrategy
    ):
        """
        method is called at start of strategyRunner.run_test
        Load the strategy to be tested.
        """
        self.my_strategy = my_strategy
        self.opp_strategy = opp_strategy

    def update_episode_n(self, ep_n: int):
        """
        method is used to sync test_manager on the iteration number that strategyRunner thinks it is on
        """
        self.ep_n = ep_n

    @abstractmethod
    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """
        method is called at start of each test episode in strategyRunner.run_test
        Reset position of robots and ball for the next strategy test.
        """
        ...

    @abstractmethod
    def eval_status(self, game: Game) -> TestStatus:
        """
        method is called on each iteration in strategyRunner.run_test
        Evaluate the status of the test episode.
        """
        ...

    @abstractmethod
    def get_n_episodes(self) -> int:
        """
        method is called at start of strategyRunner.run_test
        Get the number of episodes to run for the test.
        """
        ...
