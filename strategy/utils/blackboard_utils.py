import py_trees
from typing import Any
from strategy.common.abstract_behaviour import AbstractBehaviour

class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any, opp_strategy: bool = False):
        super().__init__(name=name, opp_strategy=opp_strategy)
        # Store the configuration, but DO NOT use the blackboard here.
        self.variable_name = variable_name
        self.value = value

    def setup(self, **kwargs):
        """
        Called once before the first tick. This is the correct place
        to get the blackboard and register keys.
        """
        super().setup(**kwargs)    
          
        self.blackboard.register_key(
            key=self.variable_name, access=py_trees.common.Access.WRITE
        )  

    def update(self) -> py_trees.common.Status:
        """
        Called every tick. Now it can safely write to the blackboard.
        """
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        print(self.blackboard.robot_id)
        return py_trees.common.Status.SUCCESS