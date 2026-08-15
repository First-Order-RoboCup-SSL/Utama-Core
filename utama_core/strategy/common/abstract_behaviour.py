from typing import Optional

import py_trees


class AbstractBehaviour(py_trees.behaviour.Behaviour):
    """An abstract base class for `strategy/referee/actions.py`'s Step classes.

    Only `setup_`/`initialise`/`update` are actually used now: `kernel.RefereeOverride`
    drives Step instances directly (`step.blackboard = shim; step.update()`),
    calling `setup_()` once but never the py_trees `setup()`/blackboard-registration
    machinery this class used to provide for real BT tree ticking — that machinery
    is gone along with the BT strategies it existed for.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        super(AbstractBehaviour, self).__init__(name)

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    def setup_(self):
        """This method is called ONCE at the end of setup(), before the first tree tick.

        The blackboard should already exist and be populated at this point.

        For adding additional blackboard keys or other setup tasks.
        """
        ...

    def initialise(self) -> None:
        """Configures and resets the behaviour ready for (repeated) execution. Initialisation is called on the first tick
        that the node is made valid ie this can be run a few times during the lifetime of a behaviour

        (DO NOT PUT EXPENSIVE OPERATIONS HERE)

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
