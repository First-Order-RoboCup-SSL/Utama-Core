from typing import Any, Optional, final

import py_trees

from config.settings import BLACKBOARD_NAMESPACE_MAP
from strategy.common.base_blackboard import BaseBlackboard


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """An abstract base class for all behaviours in the strategy."""

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        super(AbstractBehaviour, self).__init__(name)

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    def setup_(self):
        """This method is called at the end of setup(), before the first tree tick.

        For adding additional blackboard keys or other setup tasks.
        """

    def initialise(self) -> None:
        """Configures and resets the behaviour ready for (repeated) execution Initialisation is called on the first tick
        that the node is called.

        Some examples:
        - Initialising/resetting/clearing variables
        - Starting timers
        - Just-in-time discovery and establishment of middleware connections
        - Sending a goal to start a controller running elsewhere on the system
        """
        ...

    def update(self) -> py_trees.common.Status:
        """This method should be overridden by subclasses to implement the behaviour's logic.

        It should return a status indicating the outcome of the behaviour.
        """
        ...

    ### END OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    @final
    def setup(self, **kwargs: Any) -> None:
        """This method is called once by the tree before the first tick.

        We setup the common blackboard keys to all behaviours.
        """
        is_opp_strategy = kwargs.get("is_opp_strat", False)
        self.blackboard: BaseBlackboard = self.attach_blackboard_client(
            name="GlobalBlackboard", namespace=BLACKBOARD_NAMESPACE_MAP[is_opp_strategy]
        )
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
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key="tactic",
            access=py_trees.common.Access.READ,
        )
        self.setup_()

    # prevent overriding of setup method
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "setup" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override 'setup'. Override 'setup_' instead.")
