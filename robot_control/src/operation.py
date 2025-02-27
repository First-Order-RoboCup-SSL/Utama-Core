import abc
from typing import List, Dict, Optional


class Operation(metaclass=abc.ABCMeta):
    def __init__(self, game):
        self._game = game

    @abc.abstractmethod
    def enact(
        self, game, robot_ids: List[int], *args, **kwargs
    ) -> Optional[Dict[int, RobotCommand]]:  ##  {robot_id: robot_command}
        pass


class WingAttack(Operation):
    def enact(self, robot_ids: List[int], *args, **kwargs):
        # assuming centre robot has it, if close enough to goal and goal is open, shoot
        # try to pass to left or right robot if free
        # all robots move forward
        # Left robot try to pass
        robot_ids[0]

        # if done:
        #     return None
        # else:
        #     cmd1 = get_command_for_robot_1()
        #     cmd2 = get_command_for_robot_2
