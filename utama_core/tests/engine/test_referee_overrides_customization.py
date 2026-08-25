"""Tests for the strategy-implementer referee-override customization hook.

`RefereeOverride.overrides` (set via `AbstractStrategy(..., referee_overrides=...)`
or `Strategy.referee_overrides = ...`) lets a strategy replace the built-in
`*Step` formation for one specific `RefereeCommand` with its own callable,
without touching `custom_referee/actions.py` itself. This file covers:

- a registered override actually runs instead of the built-in Step
  (`test_registered_override_replaces_built_in_step`)
- an unregistered command still dispatches to its built-in Step
  (`test_unregistered_command_falls_back_to_built_in_step`)
- an override registered for a command outside `_OVERRIDE_COMMANDS` (e.g.
  `NORMAL_START`) is never invoked, since `RefereeOverride.tick()` is only
  ever reached when `is_override_command(command)` is true
  (`test_override_for_non_override_command_is_never_called`)
- `RefereeOverride.overrides` getter returns a copy, not a live view
  (`test_overrides_getter_returns_a_copy`)
- end-to-end through `AbstractStrategy`: the constructor kwarg reaches
  `robot_controller.add_robot_commands(...)` in place of the built-in
  kickoff formation (`test_abstract_strategy_end_to_end_override`)
- a goalkeeper left out of the override's cmd_map is filled in by
  `GoalkeeperTactic` rather than frozen, per `AbstractStrategy.step()`'s
  `self._goalkeeper_id not in cmd_map` gate
  (`test_goalkeeper_ticks_normally_when_override_omits_it`)

Uses the same `split_shape_runner`/`_set_referee_command` pattern as
`test_referee_override.py` for the end-to-end cases (a real rsim `Game` via
`AbstractStrategy`/`StrategyRunner`, since the override integration point —
`AbstractStrategy.load_motion_controller` assigning `Strategy.referee_overrides`
— only exists on the real objects, not a hand-built fake). The dispatcher-level
cases use a minimal fake `Game`/`MotionController` instead, since
`RefereeOverride.tick()` with a registered override never touches
`blackboard.game`'s attributes itself (the override callable owns that) —
only `game.my_team_is_yellow` is read by `RefereeOverride.tick()` before the
override lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.referee_override import RefereeOverride
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.strategy.kernel_strategy import build_split_shape_kernel_strategy

from .test_referee_override import (  # noqa: F401 (re-used fixture helper)
    _set_referee_command,
)


@dataclass
class _FakeGame:
    my_team_is_yellow: bool = True
    my_team_is_right: bool = True
    ball: Optional[object] = None
    referee: Optional[object] = None
    friendly_robots: dict = None

    def __post_init__(self):
        if self.friendly_robots is None:
            self.friendly_robots = {}


_SENTINEL_COMMAND = RobotCommand(
    local_forward_vel=1.23, local_left_vel=0.0, angular_vel=0.0, kick=False, chip=False, dribble=False
)


def test_registered_override_replaces_built_in_step():
    """A registered override for PREPARE_KICKOFF_YELLOW is called instead of
    PrepareKickoffOursStep, and its exact return value is what comes back."""
    override_calls = []

    def my_kickoff_override(game, motion_controller):
        override_calls.append((game, motion_controller))
        return {7: _SENTINEL_COMMAND}

    override = RefereeOverride(overrides={RefereeCommand.PREPARE_KICKOFF_YELLOW: my_kickoff_override})
    game = _FakeGame(my_team_is_yellow=True)
    motion_controller = Mock()

    result = override.tick(game, motion_controller, RefereeCommand.PREPARE_KICKOFF_YELLOW)

    assert result == {7: _SENTINEL_COMMAND}
    assert len(override_calls) == 1
    assert override_calls[0] == (game, motion_controller)


def test_unregistered_command_falls_back_to_built_in_step():
    """A command with no registered override still dispatches to the correct
    built-in Step — registering an override for one command must not disturb
    any other command's routing."""
    my_kickoff_override = Mock(return_value={})
    override = RefereeOverride(overrides={RefereeCommand.PREPARE_KICKOFF_YELLOW: my_kickoff_override})

    # STOP has no override registered; _step_for should still resolve it to
    # the built-in StopStep and return that Step's own computed dict (empty
    # here since _FakeGame has no friendly_robots for StopStep to iterate).
    game = _FakeGame(my_team_is_yellow=True)
    result = override.tick(game, Mock(), RefereeCommand.STOP)

    my_kickoff_override.assert_not_called()
    assert result == {}


def test_override_for_non_override_command_is_never_called():
    """An override registered for a command outside `_OVERRIDE_COMMANDS`
    (e.g. NORMAL_START) is never invoked: `Strategy.tick()` only calls into
    `RefereeOverride.tick()` at all when `is_override_command(command)` is
    true, so `RefereeOverride.tick()` itself is never even asked about
    NORMAL_START in real use. This test drives `Strategy.tick()` directly
    (rather than just `RefereeOverride.tick()`) to prove that end-to-end."""
    from utama_core.engine.context import KernelContext
    from utama_core.engine.strategy import Strategy
    from utama_core.engine.tactic import BaseTactic, TacticTag

    class _DummyTactic(BaseTactic):
        tag = TacticTag.ATTACK

        def initial_mem(self):
            return None

        def tick(self, game, ctx, robot_ids, mem):
            return {}, mem

    normal_start_override = Mock(return_value={})
    ctx = KernelContext(motion_controller=Mock())
    strategy = Strategy(
        tactics={"dummy": _DummyTactic()},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "dummy"),
        outfield_robot_ids=(1, 2, 3, 4, 5),
        ctx=ctx,
        referee_overrides={RefereeCommand.NORMAL_START: normal_start_override},
    )

    game = Mock()
    game.my_team_is_yellow = True
    referee = Mock()
    referee.referee_command = RefereeCommand.NORMAL_START
    game.referee = referee
    game.friendly_robots = {}

    strategy.tick(game)

    normal_start_override.assert_not_called()


def test_overrides_getter_returns_a_copy():
    """Mutating the dict returned by `RefereeOverride.overrides` must not
    affect subsequent `tick()` behaviour — the getter must hand back a copy,
    not the live dict backing the dispatcher."""
    my_override = Mock(return_value={})
    override = RefereeOverride(overrides={RefereeCommand.PREPARE_KICKOFF_YELLOW: my_override})

    snapshot = override.overrides
    snapshot[RefereeCommand.PREPARE_KICKOFF_YELLOW] = Mock(return_value={})
    snapshot[RefereeCommand.STOP] = Mock(return_value={})

    game = _FakeGame(my_team_is_yellow=True)
    override.tick(game, Mock(), RefereeCommand.PREPARE_KICKOFF_YELLOW)

    my_override.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end: AbstractStrategy(..., referee_overrides=...) actually reaches
# robot_controller.add_robot_commands(...), and interacts correctly with the
# goalkeeper-ticking gate in AbstractStrategy.step().
# ---------------------------------------------------------------------------


@pytest.fixture
def override_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    calls = {"n": 0}

    def kickoff_override(game, motion_controller):
        calls["n"] += 1
        # Deliberately omit the goalkeeper (id 0) and only address robot 1,
        # with a recognizable non-default command distinguishing it from
        # anything PrepareKickoffOursStep/empty_command would produce.
        return {1: _SENTINEL_COMMAND}

    strategy = AbstractStrategy(
        build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)),
        referee_overrides={RefereeCommand.PREPARE_KICKOFF_YELLOW: kickoff_override},
    )
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    runner.override_calls = calls
    real_controller = runner.my.strategy.robot_controller
    spy_controller = MagicMock(wraps=real_controller)
    runner.my.strategy.robot_controller = spy_controller
    yield runner
    runner.close()


def test_abstract_strategy_end_to_end_override(override_runner):
    """Passing `referee_overrides` to `AbstractStrategy` reaches
    `robot_controller.add_robot_commands` in place of the built-in kickoff
    formation: robot 1 receives the override's sentinel command instead of
    PrepareKickoffOursStep's ball-approach target."""
    override_runner.step_once()

    _set_referee_command(override_runner, RefereeCommand.PREPARE_KICKOFF_YELLOW)
    override_runner.step_once()

    assert override_runner.override_calls["n"] >= 1, "referee_overrides callable was never invoked"

    robot_controller = override_runner.my.strategy.robot_controller
    sent = robot_controller.add_robot_commands
    # add_robot_commands(command, robot_id) — find the call for robot 1.
    calls_for_robot_1 = [c for c in sent.call_args_list if c.args[1] == 1]
    assert calls_for_robot_1, "robot 1 never received a command this tick"
    assert calls_for_robot_1[-1].args[0] == _SENTINEL_COMMAND, "override's sentinel command did not reach robot 1"


def test_goalkeeper_ticks_normally_when_override_omits_it(override_runner):
    """The override above never addresses robot 0 (the goalkeeper). Per
    `AbstractStrategy.step()`'s `self._goalkeeper_id not in cmd_map` gate,
    GoalkeeperTactic must still tick to cover it — the goalkeeper must not be
    frozen/defaulted just because an override command is active, so long as
    the override itself doesn't claim that robot."""
    override_runner.step_once()
    game = override_runner.my.game

    # Put the goalkeeper somewhere it would have to actively move from, so a
    # passing test demonstrates GoalkeeperTactic actually ran rather than the
    # robot coincidentally staying near its spawn point.
    override_runner.sim_controller.teleport_robot(True, 0, 3.0, 3.0, 0.0)
    override_runner.step_once()
    gk_start = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)

    _set_referee_command(override_runner, RefereeCommand.PREPARE_KICKOFF_YELLOW)
    for _ in range(60):
        override_runner.step_once()

    gk_end = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)
    dist_moved = math.hypot(gk_end[0] - gk_start[0], gk_end[1] - gk_start[1])
    assert dist_moved > 0.05, "goalkeeper never moved — it was frozen instead of ticked by GoalkeeperTactic"
