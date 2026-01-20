import py_trees

from utama_core.entities.game.field import FieldBounds
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour


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
