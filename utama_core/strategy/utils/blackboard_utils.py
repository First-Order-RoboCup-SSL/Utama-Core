from typing import Any

import py_trees

from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour


class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any, opp_strategy: bool = False):
        super().__init__(name=name)
        # Store the configuration, but DO NOT use the blackboard here.
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        # print(f"Setting {self.variable_name} to {self.value} on the blackboard.")
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS
