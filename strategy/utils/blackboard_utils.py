import py_trees
from typing import Any
from strategy.common.abstract_behaviour import AbstractBehaviour

class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any, remap_to: str = None):
        super().__init__(name=name)
        # Store the configuration, but DO NOT use the blackboard here.
        self.variable_name = variable_name
        self.value = value
        self.remap_to = remap_to

    def setup(self, **kwargs):
        """
        Called once before the first tick. This is the correct place
        to get the blackboard and register keys.
        """
        super().setup(**kwargs)        

    def update(self) -> py_trees.common.Status:
        """
        Called every tick. Now it can safely write to the blackboard.
        """
        self.blackboard.register_key(
            key=self.remap_to[self.variable_name], access=py_trees.common.Access.WRITE
        )
        self.blackboard.set(self.remap_to[self.variable_name], self.value, overwrite=True)
        self.blackboard.register_key(
            key=self.remap_to[self.variable_name], access=py_trees.common.Access.READ
        )
        return py_trees.common.Status.SUCCESS