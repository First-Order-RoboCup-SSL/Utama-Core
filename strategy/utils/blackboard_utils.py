from typing import Any
import py_trees
from strategy.common.abstract_behaviour import AbstractBehaviour


class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any):
        super().__init__(name=name)
        self.variable_name = variable_name
        self.value = value
        self.blackboard.register_key(
            key=self.variable_name, access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS
