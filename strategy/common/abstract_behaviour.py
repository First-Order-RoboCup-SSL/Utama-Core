import py_trees
from abc import abstractmethod
from typing import cast
from strategy.common.base_blackboard import BaseBlackboard


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """
    An abstract base class for all behaviours in the strategy.
    """

    def __init__(self, name: str):
        super(AbstractBehaviour, self).__init__(name)
        self.blackboard: BaseBlackboard = None

    def setup(self, **kwargs):
        """
        This method is called once by the tree before the first tick.
        We get the tree from the kwargs and grab its blackboard.
        """
        tree = kwargs['tree']
        self.blackboard = tree.blackboard_client
    
    @abstractmethod
    def update(self) -> py_trees.common.Status:
        """
        This method should be overridden by subclasses to implement the behaviour's logic.
        It should return a status indicating the outcome of the behaviour.
        """
        ...
