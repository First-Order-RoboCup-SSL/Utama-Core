from utama_core.dashboard.views.referee import _serialise_state
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


def _team_info() -> TeamInfo:
    return TeamInfo(name="Test", score=0, goalkeeper=0)


def _referee_data() -> RefereeData:
    return RefereeData(
        source_identifier="test",
        time_sent=1.0,
        time_received=1.0,
        referee_command=RefereeCommand.HALT,
        referee_command_timestamp=1.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=300.0,
        blue_team=_team_info(),
        yellow_team=_team_info(),
    )


def test_serialise_state_defaults_robot_feedback_to_empty_list():
    payload = _serialise_state(_referee_data())

    assert payload["robot_feedback"] == []


def test_serialise_state_includes_robot_feedback_rows():
    robot_feedback = [
        {
            "port_id": 0,
            "has_ball": True,
            "connected": True,
            "age_seconds": 0.03,
            "team": "friendly",
            "team_color": "yellow",
            "vision_id": 2,
        }
    ]

    payload = _serialise_state(_referee_data(), robot_feedback_data=robot_feedback)

    assert payload["robot_feedback"] == robot_feedback


def test_positions_are_sent_to_the_millimetre_and_velocities_to_the_centimetre():
    from utama_core.dashboard.views.referee import _serialise_ball, _serialise_robots
    from utama_core.entities.data.vector import Vector2D, Vector3D
    from utama_core.entities.game import Ball, GameFrame, Robot

    robot = Robot(
        id=1,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(1.23456789, -0.0004999),
        v=Vector2D(0.123456, 0.0),
        a=Vector2D(0, 0),
        orientation=3.14159265,
    )
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(0.98765, 0, 0), v=Vector3D(2.34567, 0, 0), a=Vector3D(0, 0, 0)),
    )
    (r,) = _serialise_robots(frame)["friendly"]
    assert (r["x"], r["y"], r["vx"], r["orientation"]) == (1.235, -0.0, 0.12, 3.142)
    ball = _serialise_ball(frame)
    assert (ball["x"], ball["vx"]) == (0.988, 2.35)


def test_a_placement_target_must_be_inside_the_field_and_the_area_in_play():
    """Any point on the page was accepted, the run-off and the half the camera does not see
    included: the placing robots would drive out of vision."""
    from types import SimpleNamespace

    from utama_core.dashboard.views.referee import placement_target_error

    geometry = SimpleNamespace(half_length=2.0, half_width=1.5)
    camera = {"bounds": [0.0, -1.5, 2.0, 1.5]}  # the right half
    assert placement_target_error((2.0, 1.5), geometry, None) is None  # on the corner: inside
    assert "outside the field" in placement_target_error((2.01, 0.0), geometry, None)
    assert "outside the field" in placement_target_error((0.0, -1.6), geometry, camera)
    assert placement_target_error((0.0, 0.0), geometry, camera) is None  # on its edge
    assert "area in play" in placement_target_error((-0.01, 0.0), geometry, camera)


def test_a_refused_placement_is_a_bad_request_and_sends_nothing(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import pytest

    from utama_core.dashboard.views import referee as referee_view

    handlers = {}
    server = SimpleNamespace(
        add_route=lambda path, getter: None, add_command_handler=lambda path, h: handlers.update({path: h})
    )
    referee = MagicMock()
    referee.geometry = SimpleNamespace(half_length=2.0, half_width=1.5)
    referee.field_check = {"bounds": [0.0, -1.5, 2.0, 1.5]}
    monkeypatch.setattr(referee_view, "_build_static_config", lambda profile: {})
    referee_view.attach(server, referee, profile=None)
    with pytest.raises(ValueError):
        handlers["/referee/command"]({"command": "BALL_PLACEMENT_YELLOW", "designated": [-1.0, 0.0]})
    referee.force_command.assert_not_called()
    handlers["/referee/command"]({"command": "BALL_PLACEMENT_YELLOW", "designated": [1.0, 0.0]})
    assert referee.force_command.call_args.kwargs["ball_placement_target"] == (1.0, 0.0)
