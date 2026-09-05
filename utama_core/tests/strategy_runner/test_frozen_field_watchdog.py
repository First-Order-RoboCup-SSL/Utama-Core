"""Unit tests for `StrategyRunner._tick_frozen_field_watchdog` — recovery
from the rare native rsim-engine hang traced live in
replays/tournament_20260905_083358/tiki_taka_vs_zone_fluid_Lk.npz
(189s RESTART_STALL during DIRECT_FREE_BLUE).

Every robot and the ball reported exactly zero velocity (bit-for-bit, not
just "small") for the entire stall, immediately following a mid-match
`teleport_ball`-triggered `RSim.reset()`. Application-level command
computation (tactic, PID, rotation) was independently verified correct and
producing real, nonzero, correctly-directed commands throughout — the
freeze was entirely on the native engine's side and not reproducible live
from any cold-teleported starting point, so it can only be detected and
recovered from, not root-caused. See `_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS`'s
module docstring in `strategy_runner.py` for the full trace.
"""

from __future__ import annotations

from utama_core.config.enums import Mode
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.run.strategy_runner import (
    _SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS,
    StrategyRunner,
)


def _robot(rid: int, vx: float, vy: float) -> Robot:
    return Robot(
        id=rid,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(0.0, 0.0),
        v=Vector2D(vx, vy),
        a=Vector2D(0.0, 0.0),
        orientation=0.0,
    )


def _referee(command: RefereeCommand) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue", goalkeeper=0),
        yellow_team=TeamInfo(name="yellow", goalkeeper=0),
    )


def _frame(
    ts: float,
    ball_v: tuple[float, float] = (0.0, 0.0),
    robot_v: tuple[float, float] = (0.0, 0.0),
    command: RefereeCommand = RefereeCommand.FORCE_START,
) -> GameFrame:
    bvx, bvy = ball_v
    rvx, rvy = robot_v
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={0: _robot(0, rvx, rvy)},
        enemy_robots={0: _robot(0, rvx, rvy)},
        ball=Ball(p=Vector3D(1.0, 2.0, 0.0215), v=Vector3D(bvx, bvy, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
        referee=_referee(command),
    )


def _make_bare_runner() -> StrategyRunner:
    """A `StrategyRunner` with only the attributes the watchdog touches —
    bypasses `__init__` via `__new__`, same technique as
    `test_teleport_settle.py`'s `_make_bare_runner`."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    runner = StrategyRunner.__new__(StrategyRunner)
    runner.mode = Mode.RSIM
    runner.sim_controller = MagicMock()
    runner.my = SimpleNamespace(current_game_frame=None)
    runner._frozen_field_since = None
    runner.logger = MagicMock()
    return runner


def test_watchdog_recovers_after_sustained_zero_velocity():
    runner = _make_bare_runner()
    dt = 1 / 60

    t = 0.0
    ticks = int(_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS / dt) + 2
    for _ in range(ticks):
        t += dt
        runner.my.current_game_frame = _frame(t)
        runner._tick_frozen_field_watchdog()

    runner.sim_controller.teleport_ball.assert_called_once()
    call_x, call_y = runner.sim_controller.teleport_ball.call_args.args
    assert (call_x, call_y) == (1.0, 2.0)
    assert runner._frozen_field_since is None


def test_watchdog_does_not_fire_before_the_threshold():
    runner = _make_bare_runner()
    dt = 1 / 60

    t = 0.0
    short_run_ticks = int((_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS - 1.0) / dt)
    for _ in range(short_run_ticks):
        t += dt
        runner.my.current_game_frame = _frame(t)
        runner._tick_frozen_field_watchdog()

    runner.sim_controller.teleport_ball.assert_not_called()


def test_watchdog_resets_when_anything_moves():
    """A robot or the ball moving at all — even briefly — must reset the
    timer, so ordinary live play (which is never bit-exact zero for 8s
    straight) can never trip this."""
    runner = _make_bare_runner()
    dt = 1 / 60

    t = 0.0
    # Frozen for most of the window...
    almost_there = int((_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS - 0.5) / dt)
    for _ in range(almost_there):
        t += dt
        runner.my.current_game_frame = _frame(t)
        runner._tick_frozen_field_watchdog()

    # ...then genuine motion breaks the freeze.
    t += dt
    runner.my.current_game_frame = _frame(t, ball_v=(0.5, 0.0))
    runner._tick_frozen_field_watchdog()
    assert runner._frozen_field_since is None

    # Even if it freezes again immediately after, the clock restarted.
    for _ in range(almost_there):
        t += dt
        runner.my.current_game_frame = _frame(t)
        runner._tick_frozen_field_watchdog()
    runner.sim_controller.teleport_ball.assert_not_called()


def test_watchdog_ignores_halt_where_zero_velocity_is_legitimate():
    """HALT already has its own dedicated auto-resume timer, and is the one
    live-play state where the whole field is *correctly* motionless by
    design — the watchdog must not double up on it."""
    runner = _make_bare_runner()
    dt = 1 / 60

    t = 0.0
    ticks = int(_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS / dt) + 2
    for _ in range(ticks):
        t += dt
        runner.my.current_game_frame = _frame(t, command=RefereeCommand.HALT)
        runner._tick_frozen_field_watchdog()

    runner.sim_controller.teleport_ball.assert_not_called()


def test_watchdog_is_noop_outside_rsim():
    runner = _make_bare_runner()
    runner.mode = Mode.REAL
    dt = 1 / 60

    t = 0.0
    ticks = int(_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS / dt) + 2
    for _ in range(ticks):
        t += dt
        runner.my.current_game_frame = _frame(t)
        runner._tick_frozen_field_watchdog()

    runner.sim_controller.teleport_ball.assert_not_called()
