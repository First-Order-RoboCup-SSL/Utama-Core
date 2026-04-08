from types import SimpleNamespace

import numpy as np
import pytest

import utama_core.skills.src.goalkeep as gk
from utama_core.entities.data.vector import Vector2D, Vector3D

# ---------------------------------------------------------------------------
# Standard-field stub (9x6 field: goal_x = ±4.5, goal_half_width = 0.5)
# ---------------------------------------------------------------------------
_STD_LEFT_GOAL_LINE = np.array([(-4.5, 0.5), (-4.5, -0.5)])
_STD_RIGHT_GOAL_LINE = np.array([(4.5, 0.5), (4.5, -0.5)])


def _std_field(my_team_is_right: bool):
    goal_line = _STD_RIGHT_GOAL_LINE if my_team_is_right else _STD_LEFT_GOAL_LINE
    return SimpleNamespace(my_goal_line=goal_line)


# ---------------------------------------------------------------------------
# Existing tests – now with standard-field stubs
# ---------------------------------------------------------------------------


def test_intersection_with_goal_line_uses_left_goal_x():
    intersection = gk._intersection_with_goal_line(
        (-1.0, 0.0),
        (-3.0, 0.1),
        -4.5,
        0.5,
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
    intersection = gk._intersection_with_goal_line(a, b, goal_x, 0.5)

    assert intersection == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ball_pos", "defender_pos", "goal_x", "expected"),
    [
        (Vector2D(-1.0, -0.3), Vector2D(-3.0, 0.2), -4.5, -0.05),
        (Vector2D(-1.0, 0.3), Vector2D(-3.0, -0.2), -4.5, 0.05),
    ],
)
def test_single_defender_stop_y_targets_open_half(ball_pos, defender_pos, goal_x, expected):
    stop_y = gk._single_defender_stop_y(ball_pos, defender_pos, goal_x, 0.5)

    assert stop_y == pytest.approx(expected)


def test_goalkeep_fallback_uses_side_aware_shadow_target(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
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
        field=_std_field(False),
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
        field=_std_field(False),
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
        field=_std_field(False),
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


# ---------------------------------------------------------------------------
# Non-standard geometry tests
# ---------------------------------------------------------------------------


def _custom_field(goal_x: float, goal_half_width: float):
    """Create a field stub with arbitrary goal line."""
    return SimpleNamespace(
        my_goal_line=np.array([(goal_x, goal_half_width), (goal_x, -goal_half_width)]),
    )


def test_goalkeep_custom_goal_line_changes_intercept_x(monkeypatch):
    """With a wider goal at x=-6.0, the keeper should target x=-6.0."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_custom_field(-6.0, 0.8),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-5.5, 0.0)),
            1: SimpleNamespace(p=Vector2D(-4.0, 0.2)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, -0.3, 0.0),
            v=Vector3D(1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert captured["target"].x == pytest.approx(-6.0)


def test_goalkeep_custom_goal_width_changes_clamp_range(monkeypatch):
    """With goal_half_width=0.8, a prediction at y=0.7 should be kept (not clamped)."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_custom_field(-4.5, 0.8),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.0, 0.0),
            v=Vector3D(-1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: Vector2D(x, 0.7))

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    # 0.7 < 0.8 half-width, so prediction is used directly
    assert captured["target"] == Vector2D(-4.5, 0.7)


def test_goalkeep_custom_goal_width_falls_back_when_prediction_outside(monkeypatch):
    """With goal_half_width=0.3, a prediction at y=0.4 should trigger fallback."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_custom_field(-4.5, 0.3),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.0, 0.0),
            v=Vector3D(-1.0, 0.0, 0.0),
        ),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: Vector2D(x, 0.4))

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    # abs(0.4) > 0.3 half-width → fallback to stop_y=0.0
    assert captured["target"].x == pytest.approx(-4.5)
    assert captured["target"].y == pytest.approx(0.0)


def test_single_defender_stop_y_with_custom_goal_width():
    """Custom goal_half_width changes the midpoint used for shadow-based stop_y."""
    ball_pos = Vector2D(-1.0, -0.3)
    defender_pos = Vector2D(-3.0, 0.2)
    goal_x = -4.5
    # With goal_half_width=1.0 instead of 0.5
    stop_y = gk._single_defender_stop_y(ball_pos, defender_pos, goal_x, 1.0)
    # open_top=True (defender.y <= ball.y is False, so open_top=False)
    # edge_y = 0.2 - 0.1 = 0.1
    # intersection: from (-1, -0.3) to (-3, 0.1) at x=-4.5
    # dx = -2, t = (-4.5 - (-1)) / -2 = -3.5 / -2 = 1.75
    # y = -0.3 + 1.75 * (0.1 - (-0.3)) = -0.3 + 1.75 * 0.4 = -0.3 + 0.7 = 0.4
    # clamped to [-1.0, 1.0] → 0.4
    # open_top=False: (0.4 - 1.0) / 2 = -0.3
    assert stop_y == pytest.approx(-0.3)
