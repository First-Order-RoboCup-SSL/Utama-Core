"""Integration tests for the kernel model's referee-restart override.

Uses a real rsim `Game` (via `KernelStrategy`/`StrategyRunner`) rather than a
hand-built fake, since the override delegates to `strategy/referee/actions.py`
Step classes that read a wide slice of `Game` (field dims, friendly_robots,
ball, referee.designated_position, ...) — reproducing that surface with a
fake would just be a worse copy of `Game` itself. Pure-scheduling behavior
(does `Strategy.tick()` even route to the override) is covered in
`test_strategy.py`; this file checks the override actually repositions robots
legally for the two easiest-to-verify restarts.

Robots start each test placed *inside* the relevant keep-out zone
deliberately (teleported there via `sim_controller`), not left wherever the
split-shape tactics happen to put them — the split-shape formation is
naturally spread out, so a version of this test that only checked "robots
ended up far from the ball/centre" passed even when the override was not
actually engaging (caught during development: see the
`side.current_game_frame` note on `_set_referee_command` below). Starting
adversarially inside the zone means a passing test actually demonstrates the
override moved something, not that nothing needed to move.
"""

from __future__ import annotations

import math

import pytest

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.kernel.kernel_strategy import (
    KernelStrategy,
    build_split_shape_kernel_strategy,
)


@pytest.fixture
def split_shape_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = KernelStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    yield runner
    runner.close()


def _set_referee_command(runner, command: RefereeCommand, designated_position=None):
    """Inject a referee command that actually persists across ticks.

    `Game.referee` is a read-only property over `game.current` (a frozen
    `GameFrame`), so setting it needs `object.__setattr__` the same way
    `CurrentGameFrame.__init__` itself bypasses immutability. That alone is
    NOT enough, though: `StrategyRunner._run_step` rebuilds each tick's
    `GameFrame` via `dataclasses.replace(side.current_game_frame, ...)` — a
    *different* attribute than `game.current` — so a mutation applied only to
    `game.current` is silently discarded on the very next tick, reverting to
    whatever `side.current_game_frame.referee` still holds (found by tracing
    `referee_command` tick-by-tick while developing this test: it reverted to
    `None` after exactly one tick). Both attributes must be set for an
    injected command to actually hold across a multi-tick loop.
    """
    referee_data = RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue"),
        yellow_team=TeamInfo(name="yellow"),
        designated_position=designated_position,
    )
    object.__setattr__(runner.my.game.current, "referee", referee_data)
    object.__setattr__(runner.my.current_game_frame, "referee", referee_data)


def test_their_ball_placement_clears_our_robots_from_keep_out_zone(split_shape_runner):
    """During the opponent's ball placement, a robot starting inside the
    keep-out zone around the ball moves outside it — a plain tactic tick has
    no notion of this rule and would just leave it there or drive it closer."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    ball_x, ball_y = game.ball.p.x, game.ball.p.y
    split_shape_runner.sim_controller.teleport_robot(True, 1, ball_x + 0.1, ball_y, 0.0)
    split_shape_runner.step_once()  # let the teleport land in game state

    designated = (1.5, 0.0)
    _set_referee_command(split_shape_runner, RefereeCommand.BALL_PLACEMENT_BLUE, designated_position=designated)

    for _ in range(200):
        split_shape_runner.step_once()

    ball = game.ball.p
    for robot_id, robot in game.friendly_robots.items():
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        dist_to_spot = math.hypot(robot.p.x - designated[0], robot.p.y - designated[1])
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} too close to ball"
        assert dist_to_spot >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} too close to designated spot"


def test_their_kickoff_clears_our_robots_outside_center_circle(split_shape_runner):
    """A robot starting at the centre spot for the opponent's kickoff moves
    outside the keep-out radius."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    split_shape_runner.sim_controller.teleport_robot(True, 1, 0.05, 0.0, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)

    for _ in range(200):
        split_shape_runner.step_once()

    for robot_id, robot in game.friendly_robots.items():
        dist_to_center = math.hypot(robot.p.x, robot.p.y)
        assert dist_to_center >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} inside centre keep-out zone"


def test_override_ends_and_tactics_resume_on_normal_start(split_shape_runner):
    """Once NORMAL_START arrives after a restart, tactics tick again (mem was
    reset by the barrier transition on the way in) instead of the override
    continuing to hold every robot in place forever."""
    strategy = split_shape_runner.my.strategy

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)
    for _ in range(50):
        split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.NORMAL_START)
    split_shape_runner.step_once()

    partition = strategy._kernel_strategy.active_partition
    assert partition, "tactics never resumed after the restart ended"


def test_back_to_back_restarts_dispatch_a_fresh_step_each_time(split_shape_runner):
    """A second restart arriving without an intervening NORMAL_START (e.g. a
    kickoff foul immediately becoming a ball placement) must dispatch to the
    *new* command's Step, not keep re-running the first command's Step just
    because `Strategy` never saw a barrier-clearing NORMAL_START in between.

    Checked at the `RefereeOverride`/`Strategy.tick()` mechanism level
    (returned command dict differs) rather than by watching rsim physics
    settle to a final position: `_clear_to_legal_positions` recomputes its
    target from the live ball position every tick, and the ball itself can
    drift for several seconds after a restart (a known rsim convergence
    quirk unrelated to this override, see `dribble.py`'s KNOWN ISSUE note) —
    that makes "did the robot's position stop changing" an unreliable signal
    for "did the override switch commands," even though the switch itself is
    correct and instant.
    """
    game = split_shape_runner.my.game
    strategy = split_shape_runner.my.strategy._kernel_strategy
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)
    split_shape_runner.step_once()
    kickoff_commands = strategy.tick(game)

    # No NORMAL_START in between — straight into a second, different restart.
    _set_referee_command(split_shape_runner, RefereeCommand.BALL_PLACEMENT_BLUE, designated_position=(1.0, 1.0))
    split_shape_runner.step_once()
    placement_commands = strategy.tick(game)

    assert kickoff_commands != placement_commands, "override did not re-dispatch for the second restart command"


def test_goalkeeper_stops_during_halt(split_shape_runner):
    """The goalkeeper must stop issuing motion commands during HALT, same as
    the outfield pool (`Strategy.tick()`'s `is_paused` check) — it must not
    keep chasing the ball just because it's ticked outside `Strategy`."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.HALT)
    for _ in range(30):  # let residual velocity from before HALT decay first
        split_shape_runner.step_once()

    gk_pos_before = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)
    for _ in range(20):
        split_shape_runner.step_once()
    gk_pos_after = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)

    dist_moved = math.hypot(gk_pos_after[0] - gk_pos_before[0], gk_pos_after[1] - gk_pos_before[1])
    assert dist_moved < 0.05, f"goalkeeper moved {dist_moved:.3f}m while HALT was active"
