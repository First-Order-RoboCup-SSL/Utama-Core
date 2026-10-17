"""tools/restart_picture.py's capture of the referee override's commanded targets (no simulator)."""

import json
import math
from pathlib import Path

import pytest

from utama_core.analysis.plan_snapshot import build_game as build_game_with_referee
from utama_core.analysis.restart_picture import (
    ALERT,
    plan_findings,
    plan_restart,
    render_restart,
)
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee import actions
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot


def _referee(goalkeeper: int, command: RefereeCommand) -> RefereeData:
    """Referee data with our (yellow) goalkeeper set, as the replay's own team data carries it."""
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue", score=0, goalkeeper=0),
        yellow_team=TeamInfo(name="yellow", score=0, goalkeeper=goalkeeper),
    )


def _robot_plan(plan, robot_id):
    return next(r for r in plan.robots if r.robot_id == robot_id)


def test_stop_moves_a_robot_inside_keep_out_to_the_edge_and_captures_its_target():
    ball = (0.0, 0.0)
    game = build_game(
        {
            4: make_robot(4, is_friendly=True, x=0.5, y=0.0),  # 0.5 m from the ball: inside the keep-out
            5: make_robot(5, is_friendly=True, x=-3.0, y=2.0),
        },
        {0: make_robot(0, is_friendly=False, x=2.0, y=-2.0)},
        make_ball(*ball),
        my_team_is_right=True,
        my_team_is_yellow=True,
        referee_command=RefereeCommand.STOP,
    )
    real_move = actions.move

    plan = plan_restart(game, RefereeCommand.STOP)

    target = _robot_plan(plan, 4).target
    assert target is not None and _robot_plan(plan, 4).kind == "move"
    assert math.hypot(target[0] - ball[0], target[1] - ball[1]) >= BALL_KEEP_OUT_DISTANCE - 1e-6
    assert not any("keep-out" in f for f in _robot_plan(plan, 4).flags)
    assert plan.side == "both"
    assert actions.move is real_move  # the recorders are removed again after the call


def test_our_kickoff_gives_the_lowest_id_non_keeper_the_kicker_role_and_keeps_the_keeper_out():
    # Our side is yellow and defends the left goal (my_team_is_right=False): own half is x < 0.
    game = build_game_with_referee(
        {
            1: make_robot(1, is_friendly=True, x=-4.0, y=0.0),  # goalkeeper
            2: make_robot(2, is_friendly=True, x=-1.0, y=1.0),
            3: make_robot(3, is_friendly=True, x=-1.5, y=-1.0),
        },
        {0: make_robot(0, is_friendly=False, x=2.0, y=0.0)},
        make_ball(0.0, 0.0),
        my_team_is_right=False,
        my_team_is_yellow=True,
        referee=_referee(goalkeeper=1, command=RefereeCommand.PREPARE_KICKOFF_YELLOW),
    )

    plan = plan_restart(game, RefereeCommand.PREPARE_KICKOFF_YELLOW, sim_time=0.0)

    assert plan.side == "ours"
    kicker = _robot_plan(plan, 2)
    assert kicker.role == "kicker" and kicker.kind == "move"
    assert math.hypot(kicker.target[0] - 0.0, kicker.target[1]) < 0.3  # approaches the ball at centre
    assert _robot_plan(plan, 1).role == "keeper" and _robot_plan(plan, 1).kind == "none"
    assert _robot_plan(plan, 3).role is None and _robot_plan(plan, 3).kind == "move"


def test_plan_renders_a_png(tmp_path):
    game = build_game(
        {4: make_robot(4, is_friendly=True, x=0.5, y=0.0)},
        {0: make_robot(0, is_friendly=False, x=2.0, y=-2.0)},
        make_ball(0.0, 0.0),
        my_team_is_right=True,
        my_team_is_yellow=True,
        referee_command=RefereeCommand.STOP,
    )
    plan = plan_restart(game, RefereeCommand.STOP, sim_time=1.5)
    out = render_restart(plan, tmp_path / "stop.png", title="stop")

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert plan.flagged == () and ALERT.startswith("#")


def test_a_non_restart_command_is_refused():
    game = build_game(
        {4: make_robot(4, is_friendly=True, x=0.5, y=0.0)},
        {},
        make_ball(0.0, 0.0),
        referee_command=RefereeCommand.NORMAL_START,
    )
    try:
        plan_restart(game, RefereeCommand.NORMAL_START)
    except ValueError:
        return
    raise AssertionError("NORMAL_START should be refused")


# --- findings -------------------------------------------------------------------------------------


def _stop_game_with_commanded_targets(monkeypatch, targets: dict, *, my_team_is_right: bool = True):
    """A STOP game whose override commands each robot in `targets` to the given point (the step's `move` is
    what the recorder captures), so a target can sit inside the keep-out on purpose."""
    from utama_core.engine.referee_override import RefereeOverride
    from utama_core.entities.data.vector import Vector2D

    def fake_tick(self, game, motion_controller, command):
        for rid, (x, y) in targets.items():
            actions.move(game, motion_controller, rid, Vector2D(x, y), 0.0)
        return {rid: actions.empty_command(False) for rid in targets}

    monkeypatch.setattr(RefereeOverride, "tick", fake_tick)
    return build_game(
        {rid: make_robot(rid, is_friendly=True, x=2.0 + rid * 0.1, y=2.0) for rid in targets},
        {},
        make_ball(0.0, 0.0),
        my_team_is_right=my_team_is_right,
        my_team_is_yellow=True,
        referee_command=RefereeCommand.STOP,
    )


def test_targets_inside_the_keep_out_are_findings_and_the_one_cm_deadband_is_its_own_kind(monkeypatch):
    game = _stop_game_with_commanded_targets(
        monkeypatch,
        {
            7: (0.5, 0.0),  # 0.50 m from the ball: a robot sent into the circle
            8: (0.0, 0.795),  # inside the keep-out by less than 1 cm: the stop deadband
            9: (0.785, 0.0),  # 1.5 cm inside: not the deadband
        },
    )
    plan = plan_restart(game, RefereeCommand.STOP, sim_time=3.0)
    findings = plan_findings(plan, Path("match.npz"), "friendly", 3.0)

    by_robot = {f.robots[0]: f for f in findings}
    assert by_robot[7].kind == "target_in_keep_out" and by_robot[7].value == pytest.approx(0.5)
    assert by_robot[8].kind == "keep_out_short" and by_robot[8].value == pytest.approx(0.795)
    assert by_robot[9].kind == "target_in_keep_out"
    for f in findings:
        assert f.tool == "restart_picture" and f.threshold == BALL_KEEP_OUT_DISTANCE and f.unit == "m"
        assert f.side == "friendly" and f.t0 == f.t1 == 3.0 and f.replay == "match.npz"
        assert f.message.startswith(f"R{f.robots[0]} is commanded to")


def test_a_target_in_the_opponent_box_reports_how_far_it_must_move_out(monkeypatch):
    # We defend the left goal, so the opponent's box is at +x (x > 3.5 m on the standard field); a target inside
    # it, with its keep distance, is projected out, and the finding's value is that push.
    game = _stop_game_with_commanded_targets(monkeypatch, {7: (4.2, 0.0)}, my_team_is_right=False)
    (finding,) = plan_findings(plan_restart(game, RefereeCommand.STOP), Path("m.npz"), "friendly", 0.0)
    assert finding.kind == "target_in_defense_area" and finding.threshold is None
    assert finding.value > 0.0


def test_restart_occurrences_keep_every_repeat_and_restart_commands_keep_the_first(tmp_path):
    from utama_core.analysis.restart_picture import (
        restart_commands,
        restart_occurrences,
    )

    rows = [
        {"event": "referee", "sim_time": 1.0, "command": "STOP"},
        {"event": "referee", "sim_time": 2.0, "command": "NORMAL_START"},  # not a restart the override handles
        {"event": "referee", "sim_time": 5.0, "command": "STOP"},
        {"event": "referee", "sim_time": 9.0, "command": "DIRECT_FREE_BLUE"},
    ]
    replay = tmp_path / "match.npz"
    replay.write_bytes(b"")
    (tmp_path / "match.intentions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert restart_occurrences(replay) == [
        (1.0, RefereeCommand.STOP),
        (5.0, RefereeCommand.STOP),
        (9.0, RefereeCommand.DIRECT_FREE_BLUE),
    ]
    assert [t for t, _ in restart_commands(replay)] == [1.0, 9.0]
