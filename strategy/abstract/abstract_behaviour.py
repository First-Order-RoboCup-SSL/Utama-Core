import py_trees
from abc import abstractmethod


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """
    An abstract base class for all behaviours in the strategy.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        # Connect to the shared "GlobalConfig" blackboard
        self.blackboard = py_trees.blackboard.Client(name="GlobalConfig")

        # Register common keys that all behaviours might need to read
        self.blackboard.register_key(
            key="present_future_game", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="robot_controller", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="pid_oren", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="pid_trans", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="rsim_env", access=py_trees.common.Access.READ)

    @abstractmethod
    def update(self) -> py_trees.common.Status:
        """
        This method should be overridden by subclasses to implement the behaviour's logic.
        It should return a status indicating the outcome of the behaviour.
        """
        ...
