from types import SimpleNamespace

import pytest

import utama_core.skills.src.defend_parameter as dp
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D


def test_defend_parameter_uses_ball_xy_projection(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-3.8, 0.2)),
            1: SimpleNamespace(p=Vector2D(-4.0, 0.1)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.3, 0.15),
            v=Vector3D(0.0, 0.0, 0.0),
        ),
    )

    captured = {}

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        captured["dribbling"] = dribbling
        captured["robot_id"] = robot_id
        return "sentinel-command"

    monkeypatch.setattr(dp, "go_to_point", fake_go_to_point)

    result = dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    assert result == "sentinel-command"
    assert captured["robot_id"] == 1
    assert captured["dribbling"] is True
    assert isinstance(captured["target"], Vector2D)


@pytest.mark.parametrize(
    ("team_is_right", "defender_id", "ball_y", "expected_target"),
    [
        (True, 1, 0.0, Vector2D(3.0, -1.2)),
        (False, 2, -0.8, Vector2D(-3.0, 1.2)),
    ],
)
def test_defend_parameter_uses_static_assignments_for_extra_defenders(
    monkeypatch,
    team_is_right,
    defender_id,
    ball_y,
    expected_target,
):
    game = SimpleNamespace(
        my_team_is_right=team_is_right,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(0.0, 0.0)),
            1: SimpleNamespace(p=Vector2D(-1.0, 0.2)),
            2: SimpleNamespace(p=Vector2D(-1.0, -0.2)),
        },
        ball=SimpleNamespace(
            p=Vector3D(0.0, ball_y, 0.0),
            v=Vector3D(0.0, 0.0, 0.0),
        ),
    )
    captured = {}

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["robot_id"] = robot_id
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-command"

    monkeypatch.setattr(dp, "go_to_point", fake_go_to_point)

    result = dp.defend_parameter(game, motion_controller=object(), robot_id=defender_id, goal_frame_y=0.0)

    assert result == "sentinel-command"
    assert captured["robot_id"] == defender_id
    assert captured["dribbling"] is False
    assert captured["target"] == expected_target


def test_defend_parameter_trajectory_branch_clamps_to_front_of_box(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            1: SimpleNamespace(p=Vector2D(-4.0, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(0.0, 0.0, 0.0),
            v=Vector3D(-1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    def fake_predict_ball_pos_at_x(game, x):
        if x == -4.5:
            return Vector2D(-4.5, 0.2)
        if x == -4.0:
            return Vector2D(-4.0, 0.35)
        raise AssertionError(f"unexpected x lookup: {x}")

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["robot_id"] = robot_id
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-command"

    monkeypatch.setattr(dp, "predict_ball_pos_at_x", fake_predict_ball_pos_at_x)
    monkeypatch.setattr(dp, "go_to_point", fake_go_to_point)

    result = dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    assert result == "sentinel-command"
    assert captured["robot_id"] == 1
    assert captured["dribbling"] is True
    assert captured["target"].x == pytest.approx(-3.5)
    assert captured["target"].y == pytest.approx(0.1555555556)


def test_defend_parameter_offsets_final_target_by_robot_radius(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=True,
        friendly_robots={
            1: SimpleNamespace(p=Vector2D(3.0, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(0.0, 0.0, 0.0),
            v=Vector3D(0.0, 0.0, 0.0),
        ),
    )

    captured = {}

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(dp, "go_to_point", fake_go_to_point)
    monkeypatch.setattr(dp, "predict_ball_pos_at_x", lambda game, x: None)

    result = dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    assert result == "sentinel-command"
    assert captured["target"].x == pytest.approx(3.5 - ROBOT_RADIUS)
    assert captured["target"].y == pytest.approx(0.0)
