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
