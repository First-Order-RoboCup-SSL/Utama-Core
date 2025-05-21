from team_controller.src.controllers import AbstractSimController
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

    @abstractmethod
    def reset_field(self, sim_controller: AbstractSimController):
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
