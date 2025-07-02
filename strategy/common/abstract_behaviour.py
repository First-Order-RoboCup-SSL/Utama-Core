import py_trees
from abc import abstractmethod
from typing import cast
from strategy.common.base_blackboard import BaseBlackboard


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """
    An abstract base class for all behaviours in the strategy.
    """

    def __init__(self, name: str, opp_strategy: bool = False):
        super(AbstractBehaviour, self).__init__(name)
        self.unique_key = (
            "Opponent" if opp_strategy else "My"
        )
        self.blackboard: BaseBlackboard = self.attach_blackboard_client(name="GlobalBlackboard", namespace=self.unique_key)

    def setup(self, **kwargs):
        """
        This method is called once by the tree before the first tick.
        We get the tree from the kwargs and grab its blackboard.
        """
        self.blackboard.register_key(
            key="game",
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key="rsim_env",
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key="motion_controller",
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key="cmd_map",
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key="role_map",
            access=py_trees.common.Access.WRITE,
        )


    @abstractmethod
    def update(self) -> py_trees.common.Status:
        """
        This method should be overridden by subclasses to implement the behaviour's logic.
        It should return a status indicating the outcome of the behaviour.
        """
        ...
