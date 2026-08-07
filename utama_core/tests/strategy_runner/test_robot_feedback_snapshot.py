from pytest import approx

from utama_core.entities.data.command import RobotResponse
from utama_core.run.strategy_runner import (
    _build_robot_feedback_snapshot,
    _record_robot_feedback_responses,
    _RobotPortFeedbackState,
)


def test_records_raw_port_responses_and_preserves_has_ball():
    feedback = {}

    _record_robot_feedback_responses(
        feedback,
        [
            RobotResponse(id=3, has_ball=True),
            RobotResponse(id=1, has_ball=False),
        ],
        now=10.0,
    )

    snapshot = _build_robot_feedback_snapshot(
        feedback,
        now=10.1,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={},
        blue_cmd_to_vision_mapping={},
    )

    assert [row["port_id"] for row in snapshot] == [1, 3]
    assert snapshot[0]["has_ball"] is False
    assert snapshot[1]["has_ball"] is True
    assert snapshot[1]["age_seconds"] == approx(0.1)


def test_connection_flips_false_after_timeout():
    feedback = {7: _RobotPortFeedbackState(port_id=7, has_ball=True, last_seen=1.0)}

    fresh = _build_robot_feedback_snapshot(
        feedback,
        now=1.49,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={},
        blue_cmd_to_vision_mapping={},
        timeout_seconds=0.5,
    )
    stale = _build_robot_feedback_snapshot(
        feedback,
        now=1.5,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={},
        blue_cmd_to_vision_mapping={},
        timeout_seconds=0.5,
    )

    assert fresh[0]["connected"] is True
    assert stale[0]["connected"] is False


def test_mapped_ids_include_team_color_and_vision_metadata():
    feedback = {
        4: _RobotPortFeedbackState(port_id=4, has_ball=False, last_seen=20.0),
        9: _RobotPortFeedbackState(port_id=9, has_ball=True, last_seen=20.0),
    }

    snapshot = _build_robot_feedback_snapshot(
        feedback,
        now=20.0,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={4: 2},
        blue_cmd_to_vision_mapping={9: 5},
    )

    assert snapshot[0] == {
        "port_id": 4,
        "has_ball": False,
        "connected": True,
        "age_seconds": 0.0,
        "team_color": "yellow",
        "team": "friendly",
        "vision_id": 2,
    }
    assert snapshot[1] == {
        "port_id": 9,
        "has_ball": True,
        "connected": True,
        "age_seconds": 0.0,
        "team_color": "blue",
        "team": "enemy",
        "vision_id": 5,
    }


def test_mapped_robot_without_response_is_shown_as_no_data():
    snapshot = _build_robot_feedback_snapshot(
        {},
        now=30.0,
        my_team_is_yellow=True,
        yellow_cmd_to_vision_mapping={4: 2},
        blue_cmd_to_vision_mapping={},
    )

    assert snapshot == [
        {
            "port_id": 4,
            "has_ball": False,
            "connected": False,
            "age_seconds": None,
            "team_color": "yellow",
            "team": "friendly",
            "vision_id": 2,
        }
    ]


def test_unmapped_raw_responder_still_appears():
    feedback = {42: _RobotPortFeedbackState(port_id=42, has_ball=False, last_seen=4.0)}

    snapshot = _build_robot_feedback_snapshot(
        feedback,
        now=4.0,
        my_team_is_yellow=False,
        yellow_cmd_to_vision_mapping={},
        blue_cmd_to_vision_mapping={},
    )

    assert snapshot == [
        {
            "port_id": 42,
            "has_ball": False,
            "connected": True,
            "age_seconds": 0.0,
        }
    ]
