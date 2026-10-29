"""`FieldCheck`'s verdicts at their boundaries, and the runner's auto-HALT on real robots."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utama_core.config.enums import Mode
from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.custom_referee import CustomReferee
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run.field_check import (
    CAMERA_SILENT_S,
    OUT_OF_BOUNDS_MARGIN_M,
    VISION_LOST_S,
    FieldCheck,
)
from utama_core.run.strategy_runner import StrategyRunner

FIELD = FieldDimensions(
    full_field_half_length=2.0,
    full_field_half_width=1.5,
    half_defense_area_depth=0.3,
    half_defense_area_width=0.6,
    half_goal_width=0.4,
    center_circle_radius=0.3,
)
RIGHT_HALF = FieldBounds(top_left=(0.0, 1.5), bottom_right=(2.0, -1.5))


def _frame(yellow=(), blue=(), ball=None) -> RawVisionData:
    """`yellow`/`blue`: (id, x, y) per robot; `ball`: (x, y)."""
    return RawVisionData(
        ts=0.0,
        yellow_robots=[RawRobotData(i, x, y, 0.0, 1.0) for i, x, y in yellow],
        blue_robots=[RawRobotData(i, x, y, 0.0, 1.0) for i, x, y in blue],
        balls=[RawBallData(ball[0], ball[1], 0.0, 1.0)] if ball else [],
        camera_id=0,
    )


def _check(expected=(("yellow", 1),)) -> FieldCheck:
    return FieldCheck(expected, RIGHT_HALF, FIELD)


def _row(rows, name):
    return next(r for r in rows if r["name"] == name)


def test_a_configured_robot_never_seen_fails():
    fc = _check()
    fc.observe_vision([_frame(yellow=[(2, 1.0, 0.0)])], now=0.0)
    assert _row(fc.rows(0.0), "yellow 1")["status"] == "fail"


def test_a_robot_lost_by_vision_fails_only_after_the_grace_period():
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, 1.0, 0.0)])], now=0.0)
    fc.observe_vision([_frame()], now=VISION_LOST_S - 0.01)
    assert _row(fc.rows(VISION_LOST_S - 0.01), "yellow 1")["status"] == "ok"
    fc.observe_vision([_frame()], now=VISION_LOST_S + 0.01)
    assert _row(fc.rows(VISION_LOST_S + 0.01), "yellow 1")["status"] == "fail"


@pytest.mark.parametrize(
    "x, status",
    [
        (-OUT_OF_BOUNDS_MARGIN_M + 0.01, "ok"),  # reaching over the half-way line
        (-OUT_OF_BOUNDS_MARGIN_M - 0.01, "fail"),  # in the other half: wrong CAMERA_HALF, or leaving
    ],
)
def test_a_robot_outside_the_area_in_play_fails_past_the_margin(x, status):
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, x, 0.0)])], now=0.0)
    assert _row(fc.rows(0.0), "yellow 1")["status"] == status


def test_no_vision_packets_fails_the_camera():
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, 1.0, 0.0)])], now=0.0)
    assert _row(fc.rows(CAMERA_SILENT_S - 0.01), "Camera")["status"] == "ok"
    fc.observe_vision([None, None], now=CAMERA_SILENT_S + 0.01)
    assert _row(fc.rows(CAMERA_SILENT_S + 0.01), "Camera")["status"] == "fail"


def test_field_size_is_warned_until_vision_sends_it_and_fails_on_a_mismatch():
    fc = _check()
    assert _row(fc.rows(0.0), "Field size")["status"] == "warn"
    fc.observe_geometry(4000.0, 3000.0)
    assert _row(fc.rows(0.0), "Field size")["status"] == "ok"
    fc.observe_geometry(9000.0, 6000.0)
    assert _row(fc.rows(0.0), "Field size")["status"] == "fail"


def test_radio_silence_only_fails_once_the_robot_has_answered():
    """Firmware that never sends feedback must not halt every test; one that stops must."""
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, 1.0, 0.0)])], now=0.0)
    row = {"port_id": 4, "team_color": "yellow", "vision_id": 1, "has_ball": False}
    silent = dict(row, connected=False, age_seconds=None)
    assert _row(fc.rows(0.0, [silent]), "yellow 1 radio")["status"] == "warn"
    assert _row(fc.rows(0.0, [dict(row, connected=True, age_seconds=0.0)]), "yellow 1 radio")["status"] == "ok"
    lost = dict(row, connected=False, age_seconds=0.8)
    assert _row(fc.rows(0.0, [lost]), "yellow 1 radio")["status"] == "fail"


def test_an_unconfigured_robot_in_play_is_warned_and_one_outside_is_ignored():
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, 1.0, 0.0), (3, 0.5, 0.5)], blue=[(0, -1.0, 0.0)])], now=0.0)
    rows = fc.rows(0.0)
    assert _row(rows, "yellow 3")["status"] == "warn"
    assert not [r for r in rows if r["name"] == "blue 0"]


def test_halt_reason_names_the_first_failure():
    fc = _check()
    fc.observe_vision([_frame(yellow=[(1, 1.0, 0.0)], ball=(1.0, 0.0))], now=0.0)
    fc.observe_geometry(4000.0, 3000.0)
    assert fc.halt_reason(fc.rows(0.0)) is None
    fc.observe_vision([_frame(yellow=[(1, -1.0, 0.0)])], now=0.1)
    assert fc.halt_reason(fc.rows(0.1)).startswith("yellow 1:")


def _runner(mode: Mode, command: RefereeCommand) -> SimpleNamespace:
    referee = MagicMock(spec=CustomReferee)
    return SimpleNamespace(
        field_check=_check(),
        mode=mode,
        referee=referee,
        _robot_feedback_snapshot=[],
        _prev_custom_ref_command=command,
        _auto_halt=None,
        _field_check_rows=[],
        my=SimpleNamespace(current_game_frame=SimpleNamespace(ts=5.0)),
        logger=logging.getLogger("test"),
    )


@pytest.mark.parametrize(
    "mode, command, halts",
    [
        (Mode.REAL, RefereeCommand.FORCE_START, True),
        (Mode.REAL, RefereeCommand.STOP, True),  # robots move in STOP too
        (Mode.REAL, RefereeCommand.HALT, False),  # already halted: nothing to force
        (Mode.GRSIM, RefereeCommand.FORCE_START, False),  # sim resumes HALT by itself; shown only
    ],
)
def test_runner_forces_halt_on_a_failing_check_on_real_robots_only(mode, command, halts):
    runner = _runner(mode, command)
    StrategyRunner._tick_field_check(runner, [_frame()], now=0.0)  # yellow 1 never seen
    assert runner.referee.force_command.called == halts
    if halts:
        assert runner.referee.force_command.call_args.args[0] == RefereeCommand.HALT
        assert "yellow 1" in runner._auto_halt
    assert runner.referee.set_field_check.call_args.args[0]["rows"]


class _Stop(Exception):
    pass


def test_an_auto_halt_is_the_command_of_the_tick_that_forced_it():
    """Judged after the referee stepped, the tick that forced HALT still carried the
    command it replaced (FORCE_START here), so the robots got one more tick of motion."""
    from collections import deque

    command = {"now": RefereeCommand.FORCE_START}
    referee = MagicMock(spec=CustomReferee)
    referee.step.side_effect = lambda frame, t: SimpleNamespace(
        referee_command=command["now"], designated_position=None, next_command=None
    )
    referee.force_command.side_effect = lambda cmd, t, **kw: command.update(now=cmd)
    runner = _runner(Mode.REAL, RefereeCommand.FORCE_START)
    runner.referee = referee
    seen = {}

    def step_game(vision_frames, referee_data, *args, **kwargs):
        seen["command"] = referee_data.referee_command
        raise _Stop

    runner.__dict__.update(
        _vision_receiver=None,
        _draw_rsim_field_bounds_overlay=lambda: None,
        _update_robot_feedback_snapshot=lambda *a: None,
        _push_robot_feedback_to_referee=lambda: None,
        _tick_pending_placement_teleport=lambda: None,
        _tick_teleport_settle=lambda: None,
        _follow_referee_sides=lambda data: None,
        _tick_field_check=lambda frames, now: StrategyRunner._tick_field_check(runner, frames, now),
        _step_game=step_game,
        match_stats=None,
        sim_controller=None,
        ref_buffer=deque(),
        _last_referee_data=None,
        vision_buffers=[deque([_frame()])],  # yellow 1 never seen: the check fails
        game_log=None,
        opp=None,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={},
        toggle_opp_first=False,
        _halt_entered_at=None,
        _halt_resume_stop_entered_at=None,
    )
    runner.my.strategy = SimpleNamespace(robot_controller=MagicMock(**{"get_robots_responses.return_value": []}))
    with pytest.raises(_Stop):
        StrategyRunner._run_step(runner)
    assert seen["command"] == RefereeCommand.HALT


def test_the_auto_halt_reason_is_kept_through_the_halt_and_cleared_once_play_resumes():
    """It stayed for good: the dashboard's red banner was still up with the robots playing."""
    runner = _runner(Mode.REAL, RefereeCommand.FORCE_START)
    StrategyRunner._tick_field_check(runner, [_frame()], now=0.0)  # yellow 1 never seen: HALT
    runner._prev_custom_ref_command = RefereeCommand.HALT
    seen = [_frame(yellow=[(1, 1.0, 0.0)])]
    StrategyRunner._tick_field_check(runner, seen, now=0.1)
    assert runner.referee.set_field_check.call_args.args[0]["auto_halt"]
    runner._prev_custom_ref_command = RefereeCommand.STOP  # the operator resumed
    StrategyRunner._tick_field_check(runner, seen, now=0.2)
    assert runner._auto_halt is None
    assert runner.referee.set_field_check.call_args.args[0]["auto_halt"] is None


def test_the_tick_after_a_loop_stall_does_not_halt_the_next_one_does():
    """2026-10-27: the runner's loop stalled ~0.55 s every ~33 s, cameras and radio silent
    together. The first tick after judges vision before the receiver thread caught up: that
    tick must not halt on what our own stall hid, and a real loss halts on the next tick."""
    from utama_core.run.field_check import LOOP_STALL_S

    runner = _runner(Mode.REAL, RefereeCommand.FORCE_START)
    seen = _frame(yellow=[(1, 1.0, 0.0)])
    StrategyRunner._tick_field_check(runner, [seen], now=0.0)
    t = 0.0
    while t + 0.05 < VISION_LOST_S:  # normal ticks, unseen but within the grace period
        t += 0.05
        StrategyRunner._tick_field_check(runner, [None], now=t)
    stalled = t + LOOP_STALL_S + 0.1
    assert stalled > max(VISION_LOST_S, CAMERA_SILENT_S)
    StrategyRunner._tick_field_check(runner, [None], now=stalled)  # after a stall, not seen
    assert not runner.referee.force_command.called
    StrategyRunner._tick_field_check(runner, [None], now=stalled + 0.01)
    assert runner.referee.force_command.called


def test_ticks_just_under_the_stall_limit_still_halt_a_lost_robot():
    from utama_core.run.field_check import LOOP_STALL_S

    runner = _runner(Mode.REAL, RefereeCommand.FORCE_START)
    step = LOOP_STALL_S - 0.01
    StrategyRunner._tick_field_check(runner, [_frame(yellow=[(1, 1.0, 0.0)])], now=0.0)
    n = int(VISION_LOST_S / step) + 1
    for i in range(1, n + 1):
        StrategyRunner._tick_field_check(runner, [None], now=i * step)
    assert n * step > VISION_LOST_S
    assert runner.referee.force_command.called
