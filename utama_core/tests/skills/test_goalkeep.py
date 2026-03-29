from types import SimpleNamespace

import pytest

import utama_core.skills.src.goalkeep as gk
from utama_core.entities.data.vector import Vector2D, Vector3D


def test_intersection_with_goal_line_uses_left_goal_x():
    intersection = gk._intersection_with_goal_line(
        (-1.0, 0.0),
        (-3.0, 0.1),
        -4.5,
    )

    assert intersection == pytest.approx((-4.5, 0.175))


@pytest.mark.parametrize(
    ("a", "b", "goal_x", "expected"),
    [
        ((-3.0, 0.8), (-3.0, 0.9), -4.5, (-4.5, 0.5)),
        ((-1.0, 0.2), (1.0, -0.4), -4.5, (-4.5, 0.2)),
    ],
)
def test_intersection_with_goal_line_handles_vertical_and_moving_away_cases(a, b, goal_x, expected):
    intersection = gk._intersection_with_goal_line(a, b, goal_x)

    assert intersection == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ball_pos", "defender_pos", "goal_x", "expected"),
    [
        (Vector2D(-1.0, -0.3), Vector2D(-3.0, 0.2), -4.5, -0.05),
        (Vector2D(-1.0, 0.3), Vector2D(-3.0, -0.2), -4.5, 0.05),
    ],
)
def test_single_defender_stop_y_targets_open_half(ball_pos, defender_pos, goal_x, expected):
    stop_y = gk._single_defender_stop_y(ball_pos, defender_pos, goal_x)

    assert stop_y == pytest.approx(expected)


def test_goalkeep_fallback_uses_side_aware_shadow_target(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
            1: SimpleNamespace(p=Vector2D(-3.0, 0.2)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, -0.3, 0.0),
            v=Vector3D(1.0, 0.0, 0.0),
        ),
    )

    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["robot_id"] = robot_id
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-command"
    assert captured["robot_id"] == 0
    assert captured["dribbling"] is True
    assert captured["target"].x == pytest.approx(-4.5)
    assert captured["target"].y == pytest.approx(-0.05)


def test_goalkeep_uses_predicted_intercept_inside_goal(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
            1: SimpleNamespace(p=Vector2D(-3.0, 0.2)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.0, 0.0),
            v=Vector3D(-1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: Vector2D(x, 0.2))

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["robot_id"] = robot_id
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-command"
    assert captured["robot_id"] == 0
    assert captured["dribbling"] is True
    assert captured["target"] == Vector2D(-4.5, 0.2)


def test_goalkeep_three_robots_uses_midpoint_of_two_shadow_edges(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
            2: SimpleNamespace(p=Vector2D(-3.0, 0.4)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, -0.2, 0.0),
            v=Vector3D(1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-command"
    assert captured["target"].x == pytest.approx(-4.5)
    assert captured["target"].y == pytest.approx(0.4125)


def test_goalkeep_missing_expected_defender_id_falls_back_to_centre(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
            2: SimpleNamespace(p=Vector2D(-3.0, 0.2)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.0, 0.0),
            v=Vector3D(1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-command"
    assert captured["target"] == Vector2D(-4.5, 0.0)
