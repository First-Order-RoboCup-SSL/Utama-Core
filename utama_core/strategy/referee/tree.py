"""Factory for the RefereeOverride subtree.

The RefereeOverride Selector sits as the first (highest-priority) child of the root
Selector in AbstractStrategy. Each child is a Sequence:

    Sequence
    ├── CheckRefereeCommand(expected_command, ...)  ← FAILURE if no match → Selector continues
    └── <ActionStep>                                ← RUNNING while command is active

When no override matches (e.g. NORMAL_START, FORCE_START), the Selector falls through
to the user's strategy subtree.

Bilateral commands (KICKOFF / PENALTY / FREE_KICK / BALL_PLACEMENT) are split into
"ours" and "theirs" at tick-time: each action node reads my_team_is_yellow from the
game frame, so no construction-time team colour is needed.
"""

import py_trees

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.referee.actions import (
    BallPlacementOursStep,
    BallPlacementTheirsStep,
    DirectFreeOursStep,
    DirectFreeTheirsStep,
    HaltStep,
    PrepareKickoffOursStep,
    PrepareKickoffTheirsStep,
    PreparePenaltyOursStep,
    PreparePenaltyTheirsStep,
    StopStep,
)
from utama_core.strategy.referee.conditions import CheckRefereeCommand


def _make_subtree(name: str, condition: CheckRefereeCommand, action: py_trees.behaviour.Behaviour):
    """Create a Sequence([condition, action]) subtree for one referee command group."""
    seq = py_trees.composites.Sequence(name=name, memory=False)
    seq.add_children([condition, action])
    return seq


def build_referee_override_tree() -> py_trees.composites.Selector:
    """Build and return the RefereeOverride Selector.

    The returned Selector should be added as the *first* child of the root Selector
    in AbstractStrategy so that referee compliance always takes priority over strategy.

    Ours-vs-theirs resolution for bilateral commands:
    Each pair of action nodes (e.g. BallPlacementOursStep / BallPlacementTheirsStep)
    reads my_team_is_yellow from the game frame at tick-time to determine which role
    to play. The CheckRefereeCommand condition simply checks which specific command
    (YELLOW or BLUE variant) is active; the action node then maps that to our/their role.

    Priority order (top = highest):
      1. HALT           — immediate stop, no exceptions
      2. STOP           — slowed stop, keep distance from ball
      3. TIMEOUT        — idle (same as STOP)
      4. BALL_PLACEMENT — ours or theirs
      5. PREPARE_KICKOFF
      6. PREPARE_PENALTY
      7. DIRECT_FREE
    """
    override = py_trees.composites.Selector(name="RefereeOverride", memory=False)

    # 1. HALT
    override.add_child(
        _make_subtree(
            "Halt",
            CheckRefereeCommand(RefereeCommand.HALT),
            HaltStep(name="HaltStep"),
        )
    )

    # 2. STOP
    override.add_child(
        _make_subtree(
            "Stop",
            CheckRefereeCommand(RefereeCommand.STOP),
            StopStep(name="StopStep"),
        )
    )

    # 3. TIMEOUT (yellow or blue — same behaviour: idle)
    override.add_child(
        _make_subtree(
            "Timeout",
            CheckRefereeCommand(RefereeCommand.TIMEOUT_YELLOW, RefereeCommand.TIMEOUT_BLUE),
            StopStep(name="TimeoutStop"),
        )
    )

    # 4a. BALL_PLACEMENT — yellow team places ball
    override.add_child(
        _make_subtree(
            "BallPlacementYellow",
            CheckRefereeCommand(RefereeCommand.BALL_PLACEMENT_YELLOW),
            _BallPlacementDispatch(is_yellow_command=True, name="BallPlacementYellowStep"),
        )
    )

    # 4b. BALL_PLACEMENT — blue team places ball
    override.add_child(
        _make_subtree(
            "BallPlacementBlue",
            CheckRefereeCommand(RefereeCommand.BALL_PLACEMENT_BLUE),
            _BallPlacementDispatch(is_yellow_command=False, name="BallPlacementBlueStep"),
        )
    )

    # 5a. PREPARE_KICKOFF — yellow team kicks off
    override.add_child(
        _make_subtree(
            "KickoffYellow",
            CheckRefereeCommand(RefereeCommand.PREPARE_KICKOFF_YELLOW),
            _KickoffDispatch(is_yellow_command=True, name="KickoffYellowStep"),
        )
    )

    # 5b. PREPARE_KICKOFF — blue team kicks off
    override.add_child(
        _make_subtree(
            "KickoffBlue",
            CheckRefereeCommand(RefereeCommand.PREPARE_KICKOFF_BLUE),
            _KickoffDispatch(is_yellow_command=False, name="KickoffBlueStep"),
        )
    )

    # 6a. PREPARE_PENALTY — yellow team takes penalty
    override.add_child(
        _make_subtree(
            "PenaltyYellow",
            CheckRefereeCommand(RefereeCommand.PREPARE_PENALTY_YELLOW),
            _PenaltyDispatch(is_yellow_command=True, name="PenaltyYellowStep"),
        )
    )

    # 6b. PREPARE_PENALTY — blue team takes penalty
    override.add_child(
        _make_subtree(
            "PenaltyBlue",
            CheckRefereeCommand(RefereeCommand.PREPARE_PENALTY_BLUE),
            _PenaltyDispatch(is_yellow_command=False, name="PenaltyBlueStep"),
        )
    )

    # 7a. DIRECT_FREE — yellow team's free kick
    override.add_child(
        _make_subtree(
            "DirectFreeYellow",
            CheckRefereeCommand(RefereeCommand.DIRECT_FREE_YELLOW),
            _DirectFreeDispatch(is_yellow_command=True, name="DirectFreeYellowStep"),
        )
    )

    # 7b. DIRECT_FREE — blue team's free kick
    override.add_child(
        _make_subtree(
            "DirectFreeBlue",
            CheckRefereeCommand(RefereeCommand.DIRECT_FREE_BLUE),
            _DirectFreeDispatch(is_yellow_command=False, name="DirectFreeBlueStep"),
        )
    )

    return override


# ---------------------------------------------------------------------------
# Dispatcher nodes
#
# Each dispatcher reads my_team_is_yellow from the game frame at tick-time
# and delegates to the correct Ours/Theirs action node.
#
# Using separate Ours/Theirs classes directly (rather than conditionals in
# a single node) keeps each action node's logic clean and single-purpose.
# The dispatcher is a thin routing layer that composes them.
# ---------------------------------------------------------------------------

from utama_core.strategy.common.abstract_behaviour import (  # noqa: E402
    AbstractBehaviour,
)


class _BallPlacementDispatch(AbstractBehaviour):
    """Routes to BallPlacementOursStep or BallPlacementTheirsStep at tick-time."""

    def __init__(self, is_yellow_command: bool, name: str):
        super().__init__(name=name)
        self._is_yellow_command = is_yellow_command
        self._ours = BallPlacementOursStep(name="BallPlacementOurs")
        self._theirs = BallPlacementTheirsStep(name="BallPlacementTheirs")

    def setup_(self):
        # Propagate setup to the inner nodes so their blackboards are initialised
        self._ours.setup(is_opp_strat=False)
        self._theirs.setup(is_opp_strat=False)

    def update(self) -> py_trees.common.Status:
        if self._is_yellow_command == self.blackboard.game.my_team_is_yellow:
            self._ours.blackboard = self.blackboard
            return self._ours.update()
        else:
            self._theirs.blackboard = self.blackboard
            return self._theirs.update()


class _KickoffDispatch(AbstractBehaviour):
    """Routes to PrepareKickoffOursStep or PrepareKickoffTheirsStep at tick-time."""

    def __init__(self, is_yellow_command: bool, name: str):
        super().__init__(name=name)
        self._is_yellow_command = is_yellow_command
        self._ours = PrepareKickoffOursStep(name="KickoffOurs")
        self._theirs = PrepareKickoffTheirsStep(name="KickoffTheirs")

    def setup_(self):
        self._ours.setup(is_opp_strat=False)
        self._theirs.setup(is_opp_strat=False)

    def update(self) -> py_trees.common.Status:
        if self._is_yellow_command == self.blackboard.game.my_team_is_yellow:
            self._ours.blackboard = self.blackboard
            return self._ours.update()
        else:
            self._theirs.blackboard = self.blackboard
            return self._theirs.update()


class _PenaltyDispatch(AbstractBehaviour):
    """Routes to PreparePenaltyOursStep or PreparePenaltyTheirsStep at tick-time."""

    def __init__(self, is_yellow_command: bool, name: str):
        super().__init__(name=name)
        self._is_yellow_command = is_yellow_command
        self._ours = PreparePenaltyOursStep(name="PenaltyOurs")
        self._theirs = PreparePenaltyTheirsStep(name="PenaltyTheirs")

    def setup_(self):
        self._ours.setup(is_opp_strat=False)
        self._theirs.setup(is_opp_strat=False)

    def update(self) -> py_trees.common.Status:
        if self._is_yellow_command == self.blackboard.game.my_team_is_yellow:
            self._ours.blackboard = self.blackboard
            return self._ours.update()
        else:
            self._theirs.blackboard = self.blackboard
            return self._theirs.update()


class _DirectFreeDispatch(AbstractBehaviour):
    """Routes to DirectFreeOursStep or DirectFreeTheirsStep at tick-time."""

    def __init__(self, is_yellow_command: bool, name: str):
        super().__init__(name=name)
        self._is_yellow_command = is_yellow_command
        self._ours = DirectFreeOursStep(name="DirectFreeOurs")
        self._theirs = DirectFreeTheirsStep(name="DirectFreeTheirs")

    def setup_(self):
        self._ours.setup(is_opp_strat=False)
        self._theirs.setup(is_opp_strat=False)

    def update(self) -> py_trees.common.Status:
        if self._is_yellow_command == self.blackboard.game.my_team_is_yellow:
            self._ours.blackboard = self.blackboard
            return self._ours.update()
        else:
            self._theirs.blackboard = self.blackboard
            return self._theirs.update()
