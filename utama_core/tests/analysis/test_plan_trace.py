"""tools/plan_trace.py's trace function on synthetic games over time (no replay, no simulator)."""

import math

import pytest

from utama_core.analysis.plan_trace import (
    PARKED_MIN_S,
    TraceRow,
    _referee_spans,
    carrot_kind,
)
from utama_core.analysis.plan_trace import plan_findings as plan_trace_findings
from utama_core.analysis.plan_trace import plan_trace, sample_times, short_carrot_label
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


# --- findings -------------------------------------------------------------------------------------


def _short_row(t: float, *, robot_to_ball: float, carrot_to_ball: float, robot_to_target: float = 2.0):
    """A short-carrot tick: the carrot collapsed onto the robot (0.05 m) while the target is 2 m away.
    The ball is at the origin, the robot on the x axis at `robot_to_ball`, the carrot at `carrot_to_ball`."""
    return TraceRow(
        t=t,
        robot=(robot_to_ball, 0.0),
        target=(3.0, 0.0),
        carrot=(carrot_to_ball, 0.0),
        robot_to_carrot=0.05,
        robot_to_target=robot_to_target,
        carrot_offset=0.0,
        robot_to_ball=robot_to_ball,
        clearance=1.0,
        speed=0.0,
        ball=(0.0, 0.0),
        referee="NORMAL_START",
    )


def _far_row(t: float):
    """A tick whose carrot is well clear of the robot: not a short carrot, so it breaks a run."""
    return TraceRow(
        t=t,
        robot=(2.0, 0.0),
        target=(3.0, 0.0),
        carrot=(2.5, 0.0),
        robot_to_carrot=0.5,
        robot_to_target=1.0,
        carrot_offset=0.0,
        robot_to_ball=2.0,
        clearance=1.0,
        speed=0.5,
        ball=(0.0, 0.0),
        referee="NORMAL_START",
    )


def test_pushed_out_is_a_short_carrot_that_starts_inside_the_keep_out_and_ends_outside():
    assert carrot_kind(_short_row(0.0, robot_to_ball=0.79, carrot_to_ball=0.9)) == "pushed_out"
    assert carrot_kind(_short_row(0.0, robot_to_ball=0.5, carrot_to_ball=0.9)) == "pushed_out"


def test_parked_is_a_short_carrot_that_is_not_pushed_across_the_keep_out_edge():
    # Robot exactly on the 0.8 m edge is not inside it (strict inequality): parked.
    assert carrot_kind(_short_row(0.0, robot_to_ball=0.8, carrot_to_ball=0.9)) == "parked"
    # Robot already outside, carrot also outside: parked.
    assert carrot_kind(_short_row(0.0, robot_to_ball=0.81, carrot_to_ball=0.9)) == "parked"
    # Robot inside and carrot inside too: nothing was pushed out, so parked.
    assert carrot_kind(_short_row(0.0, robot_to_ball=0.79, carrot_to_ball=0.7)) == "parked"


def test_a_run_of_one_second_or_more_is_one_finding_per_span():
    pushed = [_short_row(round(i * 0.1, 6), robot_to_ball=0.7, carrot_to_ball=0.9) for i in range(11)]  # 1.0 s
    parked = [_short_row(round((11 + i) * 0.1, 6), robot_to_ball=1.5, carrot_to_ball=1.6) for i in range(11)]
    findings = plan_trace_findings(pushed + parked, 4, replay="x.npz", side="friendly")

    assert [f.kind for f in findings] == ["pushed_out", "parked"]
    assert findings[0].robots == (4,) and findings[0].value == 1.0 and findings[0].threshold == PARKED_MIN_S
    assert findings[0].t0 == 0.0 and findings[0].t1 == pytest.approx(1.0)
    assert findings[1].t0 == pytest.approx(1.1)


def test_a_run_under_a_second_and_a_gap_inside_a_run_give_no_finding():
    short = [_short_row(round(i * 0.1, 6), robot_to_ball=1.5, carrot_to_ball=1.6) for i in range(9)]  # 0.8 s
    gap = [_far_row(0.9)]
    rows = short + gap + [_short_row(round(1.0 + i * 0.1, 6), robot_to_ball=1.5, carrot_to_ball=1.6) for i in range(9)]
    assert plan_trace_findings(rows, 4, replay=None, side="friendly") == []
