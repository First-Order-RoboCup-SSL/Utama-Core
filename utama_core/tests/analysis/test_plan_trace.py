"""tools/plan_trace.py's trace function on synthetic games over time (no replay, no simulator)."""

import math

import pytest

from utama_core.analysis.plan_trace import (
    TraceRow,
    _referee_spans,
    plan_trace,
    sample_times,
    short_carrot_label,
)
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

BALL = (0.0, -1.5)
ENEMY = (2.0, 1.5)


def _game(t: float, referee: RefereeCommand = RefereeCommand.NORMAL_START):
    """Our robot 4 drives along the x axis at 0.5 m/s from x = -1; the rest of the field is still."""
    return build_game(
        {4: make_robot(4, is_friendly=True, x=-1.0 + 0.5 * t, y=0.0)},
        {0: make_robot(0, is_friendly=False, x=ENEMY[0], y=ENEMY[1])},
        make_ball(*BALL),
        my_team_is_right=True,
        my_team_is_yellow=True,
        referee_command=referee,
        ts=t,
    )


def test_sample_times_keeps_the_end_even_off_the_grid():
    assert sample_times(0.0, 0.2, 0.1) == [0.0, 0.1, 0.2]
    assert sample_times(0.0, 0.25, 0.1) == [0.0, 0.1, 0.2, 0.25]
    assert sample_times(3.0, 3.0, 0.1) == [3.0]
    with pytest.raises(ValueError):
        sample_times(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        sample_times(1.0, 0.0, 0.1)


def test_trace_rows_and_png_for_a_fixed_target(tmp_path):
    out = tmp_path / "trace.png"

    trace = plan_trace(_game, 4, 0.0, 0.2, out, step=0.1, target=(1.0, 0.0), title="synthetic")

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert trace.path == out
    assert [r.t for r in trace.rows] == [0.0, 0.1, 0.2]
    assert trace.target_note.startswith("target: (1.00, 0.00) fixed")
    for r in trace.rows:
        assert r.robot == pytest.approx((-1.0 + 0.5 * r.t, 0.0))
        assert r.target == (1.0, 0.0)
        assert r.robot_to_ball == pytest.approx(math.dist(r.robot, BALL))
        assert r.robot_to_carrot == pytest.approx(math.dist(r.robot, r.carrot))
        assert r.robot_to_target == pytest.approx(math.dist(r.robot, r.target))
        assert r.carrot_offset >= 0.0
        assert r.referee == "NORMAL_START"
        assert r.ball == BALL
    assert trace.rows[0].speed is None
    assert trace.rows[1].speed == pytest.approx(0.5)
    assert trace.rows[2].speed == pytest.approx(0.5)


def test_stand_in_target_is_the_robot_lookahead_seconds_later(tmp_path):
    trace = plan_trace(_game, 4, 0.0, 0.2, tmp_path / "standin.png", step=0.1, lookahead=0.3)

    assert trace.target_note == "target: stand-in, robot at t+0.3 s"
    for r in trace.rows:
        assert r.target == pytest.approx((-1.0 + 0.5 * (r.t + 0.3), 0.0))


def test_referee_changes_show_as_separate_spans(tmp_path):
    def game_at(t):
        command = RefereeCommand.STOP if t < 0.15 else RefereeCommand.NORMAL_START
        return _game(t, command)

    trace = plan_trace(game_at, 4, 0.0, 0.3, tmp_path / "ref.png", step=0.1, target=(1.0, 0.0))

    assert [r.referee for r in trace.rows] == ["STOP", "STOP", "NORMAL_START", "NORMAL_START"]
    spans = _referee_spans(list(trace.rows), 0.3)
    assert [(name, round(start, 3), round(end, 3)) for name, start, end in spans] == [
        ("STOP", 0.0, 0.2),
        ("NORMAL_START", 0.2, 0.3),
    ]


def _row(carrot, robot=(0.762, -0.225), ball=(0.0, 0.0)):
    return TraceRow(
        t=258.0,
        robot=robot,
        target=(-1.0, -0.3),
        carrot=carrot,
        robot_to_carrot=math.dist(robot, carrot),
        robot_to_target=math.dist(robot, (-1.0, -0.3)),
        carrot_offset=0.0,
        robot_to_ball=math.dist(robot, ball),
        clearance=0.216,
        speed=None,
        ball=ball,
        referee="PREPARE_KICKOFF_YELLOW",
    )


def test_a_short_carrot_out_past_the_keep_out_is_a_push_out_not_a_parked_robot():
    """counter_flow_vs_tiki_taka_plus at 258 s: robot 0.795 m from the ball, carrot 0.85 m from it."""
    assert "pushed out of the keep-out" in short_carrot_label(_row(carrot=(0.811, -0.236)))
    # The same short carrot left inside the circle is a stall.
    assert "parked" in short_carrot_label(_row(carrot=(0.75, -0.22)))
