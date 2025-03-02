import py_trees

from strategy.behaviour_trees.behaviours.robocup_behaviour import RobocupBehaviour


class DummyBehaviour(RobocupBehaviour):
    def __init__(self):
        super().__init__(name="TestBT")
        print("%s.__init__()" % (self.__class__.__name__))

    def update(self) -> py_trees.common.Status:
        print("%s.update()" % (self.__class__.__name__))
        print(self.blackboard.robot_controller)
        return py_trees.common.Status.SUCCESS
