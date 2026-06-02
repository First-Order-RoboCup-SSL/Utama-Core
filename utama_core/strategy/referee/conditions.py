from typing import Tuple

import py_trees

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour


class CheckRefereeCommand(AbstractBehaviour):
    """Returns SUCCESS if the current referee command matches any of the given expected commands.

    Returns FAILURE if there is no referee data, or if the command does not match.
    Used as the first child of each referee subtree's Sequence so the Sequence fails fast
    and the parent Selector moves on to the next subtree.

    Args:
        expected_commands: One or more RefereeCommand values to match against.
    """

    def __init__(self, *expected_commands: RefereeCommand):
        name = "CheckCmd?" + "|".join(c.name for c in expected_commands)
        super().__init__(name=name)
        self.expected_commands: Tuple[RefereeCommand, ...] = expected_commands

    def update(self) -> py_trees.common.Status:
        ref = self.blackboard.game.referee
        if ref is None:
            return py_trees.common.Status.FAILURE
        if ref.referee_command in self.expected_commands:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
