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
