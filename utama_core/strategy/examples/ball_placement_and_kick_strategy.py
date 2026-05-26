import py_trees

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.common import AbstractBehaviour
from utama_core.strategy.examples.ball_placement_strategy import BallPlacementStrategy


class KickAfterDirectFreeStep(AbstractBehaviour):
    """Kick once on the first NORMAL_START after a referee-managed direct free."""

    _KICK_DISTANCE = 0.22

    def setup_(self):
        self._seen_first_live_restart = False
        self._active_restart_timestamp: float | None = None
        self._kick_sent_for_timestamp: float | None = None

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        ref = game.referee

        if ref is None or ref.referee_command != RefereeCommand.NORMAL_START:
            return py_trees.common.Status.RUNNING

        restart_timestamp = ref.referee_command_timestamp
        if not self._seen_first_live_restart:
            self._seen_first_live_restart = True
            self._active_restart_timestamp = restart_timestamp
            return py_trees.common.Status.RUNNING

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

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="PlacementRestartKickRoot", memory=False)
        root.add_child(KickAfterDirectFreeStep(name="KickAfterDirectFree"))
        return root
