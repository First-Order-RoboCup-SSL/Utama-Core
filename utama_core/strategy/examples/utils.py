from typing import Any

import py_trees

from utama_core.entities.game import FieldBounds
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour


class SetBlackboardVariable(AbstractBehaviour):
    """
    Writes a constant `value` onto the blackboard with the key `variable_name`.
    **Blackboard Interaction:**
        - Writes:
            - `variable_name` (Any): The name of the blackboard variable to be set.
    **Returns:**
        - `py_trees.common.Status.SUCCESS`: The variable has been set.
    """

    def __init__(self, name: str, variable_name: str, value: Any):
        super().__init__(name=name)
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS


class CalculateFieldCenter(AbstractBehaviour):
    """
    Calculates the center of the provided field bounds and writes it to the blackboard.
    """

    def __init__(self, field_bounds: FieldBounds, output_key: str = "FieldCenter"):
        super().__init__(name="CalculateFieldCenter")
        self.output_key = output_key
        self.field_bounds = field_bounds
        self.calculated = False

    def setup_(self):
        self.blackboard.register_key(key=self.output_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.calculated:
            return py_trees.common.Status.SUCCESS

        center = self.field_bounds.center
        self.blackboard.set(self.output_key, center, overwrite=True)
        self.calculated = True
        return py_trees.common.Status.SUCCESS
