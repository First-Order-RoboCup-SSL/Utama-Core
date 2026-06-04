from typing import Optional

import py_trees

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.common import AbstractBehaviour
from utama_core.strategy.examples.ball_placement_strategy import BallPlacementStrategy

_DIRECT_FREE_COMMANDS = frozenset(
    {
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.DIRECT_FREE_BLUE,
    }
)


class _CommandTracker:
    """Mutable container updated by BallPlacementAndKickStrategy.step() every tick.

    Because KickAfterDirectFreeStep is not ticked while the referee-override tree
    handles DIRECT_FREE_* commands, it cannot observe that command transition
    directly.  The strategy's step() method samples the referee command before
    each tree tick and records it here so the behaviour can read it on the next
    NORMAL_START tick.
    """

    def __init__(self):
        self.prev_command: Optional[RefereeCommand] = None


class KickAfterDirectFreeStep(AbstractBehaviour):
    """Kick once on each NORMAL_START that follows a direct-free command."""

    _KICK_DISTANCE = 0.22

    def __init__(self, name: str, tracker: _CommandTracker):
        super().__init__(name=name)
        self._tracker = tracker

    def setup_(self):
        self._active_restart_timestamp: float | None = None
        self._kick_sent_for_timestamp: float | None = None

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee

        if ref is None:
            return py_trees.common.Status.RUNNING

        current_command = ref.referee_command

        if current_command != RefereeCommand.NORMAL_START:
            return py_trees.common.Status.RUNNING

        # Only kick if we transitioned here from a direct free (not from a kickoff
        # seed / force-start at game start).  _tracker.prev_command is maintained
        # by the strategy's step() so it reflects commands seen during ticks when
        # this node was overridden by the RefereeOverride subtree.
        if self._tracker.prev_command not in _DIRECT_FREE_COMMANDS:
            return py_trees.common.Status.RUNNING

        restart_timestamp = ref.referee_command_timestamp
        if restart_timestamp != self._active_restart_timestamp:
            self._active_restart_timestamp = restart_timestamp
            self._kick_sent_for_timestamp = None

        if self._kick_sent_for_timestamp == restart_timestamp:
            return py_trees.common.Status.RUNNING

        if game.ball is None or not game.friendly_robots:
            return py_trees.common.Status.RUNNING

        kicker_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p),
        )
        kicker = game.friendly_robots[kicker_id]
        if kicker.p.distance_to(game.ball.p) > self._KICK_DISTANCE:
            return py_trees.common.Status.RUNNING

        self.blackboard.cmd_map[kicker_id] = RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            kick=1,
            chip=0,
            dribble=0,
        )
        self._kick_sent_for_timestamp = restart_timestamp

        return py_trees.common.Status.RUNNING


class BallPlacementAndKickStrategy(BallPlacementStrategy):
    """Use referee ball placement/direct-free setup, then kick once on NORMAL_START."""

    def __init__(self):
        self._tracker = _CommandTracker()
        super().__init__()

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="PlacementRestartKickRoot", memory=False)
        root.add_child(KickAfterDirectFreeStep(name="KickAfterDirectFree", tracker=self._tracker))
        return root

    def step(self):
        # Sample the referee command *before* the tree tick so that transitions
        # through DIRECT_FREE_* (which are handled by the RefereeOverride subtree
        # and never reach KickAfterDirectFreeStep) are still recorded in the tracker.
        game = self.blackboard.game
        ref = game.referee
        if ref is not None:
            self._tracker.prev_command = ref.referee_command
        super().step()
