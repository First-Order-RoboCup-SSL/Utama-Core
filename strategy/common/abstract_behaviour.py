import py_trees
from abc import abstractmethod
from typing import cast
from strategy.common.base_blackboard import BaseBlackboard


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """
    An abstract base class for all behaviours in the strategy.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        # Connect to the shared "GlobalConfig" blackboard
        client_bb = py_trees.blackboard.Client(name="GlobalConfig")

        # Register common keys that all behaviours might need to read
        client_bb.register_key(key="game", access=py_trees.common.Access.READ)
        client_bb.register_key(
            key="robot_controller", access=py_trees.common.Access.READ
        )
        client_bb.register_key(
            key="motion_controller", access=py_trees.common.Access.READ
        )
        client_bb.register_key(key="rsim_env", access=py_trees.common.Access.READ)
        client_bb.register_key(key="cmd_map", access=py_trees.common.Access.WRITE)
        client_bb.register_key(key="role_map", access=py_trees.common.Access.WRITE)
        self.blackboard: BaseBlackboard = cast(BaseBlackboard, client_bb)

    @abstractmethod
    def update(self) -> py_trees.common.Status:
        """
        This method should be overridden by subclasses to implement the behaviour's logic.
        It should return a status indicating the outcome of the behaviour.
        """
        ...
