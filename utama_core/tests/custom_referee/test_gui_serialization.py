import json

from utama_core.custom_referee.gui import _serialise_state
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
    payload = json.loads(_serialise_state(_referee_data()))

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

    payload = json.loads(_serialise_state(_referee_data(), robot_feedback_data=robot_feedback))

    assert payload["robot_feedback"] == robot_feedback
