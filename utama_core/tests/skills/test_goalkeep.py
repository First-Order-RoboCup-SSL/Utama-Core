from types import SimpleNamespace

import numpy as np
import pytest

import utama_core.skills.src.goalkeep as gk
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.skills.src.utils.defense_utils import (
    intersection_with_x_line,
    single_defender_stop_y,
)

EDGE_OFFSET = BALL_RADIUS + ROBOT_RADIUS

# ---------------------------------------------------------------------------
# Standard-field stub (9x6 field: goal_x = +/-4.5, goal_half_width = 0.5)
# ---------------------------------------------------------------------------
_STD_LEFT_GOAL_LINE = np.array([(-4.5, 0.5), (-4.5, -0.5)])
_STD_RIGHT_GOAL_LINE = np.array([(4.5, 0.5), (4.5, -0.5)])

_LEFT_KEEPER_X = -4.5 + ROBOT_RADIUS
_RIGHT_KEEPER_X = 4.5 - ROBOT_RADIUS
_STD_POST_LIMIT = 0.5 - ROBOT_RADIUS


def _std_field(my_team_is_right: bool):
    goal_line = _STD_RIGHT_GOAL_LINE if my_team_is_right else _STD_LEFT_GOAL_LINE
    return SimpleNamespace(my_goal_line=goal_line, half_goal_width=0.5)


# ---------------------------------------------------------------------------
# Shared helper unit tests (intersection_with_x_line)
# ---------------------------------------------------------------------------


def test_intersection_with_x_line_basic():
    intersection = intersection_with_x_line(
        (-1.0, 0.0),
        (-3.0, 0.1),
        -4.5,
        0.5,
    )

    assert intersection == pytest.approx((-4.5, 0.175))


@pytest.mark.parametrize(
    ("a", "b", "target_x", "expected"),
    [
        ((-3.0, 0.8), (-3.0, 0.9), -4.5, (-4.5, 0.5)),
        ((-1.0, 0.2), (1.0, -0.4), -4.5, (-4.5, 0.2)),
    ],
)
def test_intersection_with_x_line_handles_vertical_and_moving_away_cases(a, b, target_x, expected):
    intersection = intersection_with_x_line(a, b, target_x, 0.5)

    assert intersection == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Shared helper unit tests (single_defender_stop_y)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ball_pos", "defender_pos", "keeper_x", "post_limit", "expected_sign"),
    [
        # Defender shifted positive-y relative to ball → keeper covers negative gap
        (Vector2D(-1.0, -0.3), Vector2D(-3.0, 0.2), -4.5, 0.5, -1),
        # Mirror: defender shifted negative-y → keeper covers positive gap
        (Vector2D(-1.0, 0.3), Vector2D(-3.0, -0.2), -4.5, 0.5, 1),
    ],
)
def test_single_defender_stop_y_targets_open_half(ball_pos, defender_pos, keeper_x, post_limit, expected_sign):
    stop_y = single_defender_stop_y(ball_pos, defender_pos, keeper_x, post_limit, EDGE_OFFSET)

    # The keeper should be on the opposite side of the defender's offset
    assert stop_y * expected_sign > 0


def test_single_defender_stop_y_with_custom_post_limit():
    """Wider post_limit changes the midpoint used for shadow-based stop_y."""
    ball_pos = Vector2D(-1.0, -0.3)
    defender_pos = Vector2D(-3.0, 0.2)
    keeper_x = -4.5
    narrow = single_defender_stop_y(ball_pos, defender_pos, keeper_x, 0.5, EDGE_OFFSET)
    wide = single_defender_stop_y(ball_pos, defender_pos, keeper_x, 1.0, EDGE_OFFSET)
    # Wider post_limit means larger uncovered region → keeper shifts further
    assert abs(wide) > abs(narrow)


# ---------------------------------------------------------------------------
# Goalkeep integration tests
# ---------------------------------------------------------------------------


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
    assert captured["target"].x == pytest.approx(_LEFT_KEEPER_X)
    # Keeper should be in the negative-y gap (defender is at y=0.2, ball at y=-0.3)
    assert captured["target"].y < 0


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
    assert captured["target"] == Vector2D(_LEFT_KEEPER_X, 0.2)


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
    assert captured["target"].x == pytest.approx(_LEFT_KEEPER_X)
    # Compute expected: intersection of two shadow edges at keeper_x with post_limit clamp
    _, yy1 = intersection_with_x_line(
        (-1.0, -0.2),
        (-3.0, 0.0 + EDGE_OFFSET),
        _LEFT_KEEPER_X,
        _STD_POST_LIMIT,
    )
    _, yy2 = intersection_with_x_line(
        (-1.0, -0.2),
        (-3.0, 0.4 - EDGE_OFFSET),
        _LEFT_KEEPER_X,
        _STD_POST_LIMIT,
    )
    assert captured["target"].y == pytest.approx((yy1 + yy2) / 2)


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
    assert captured["target"] == Vector2D(_LEFT_KEEPER_X, 0.0)


# ---------------------------------------------------------------------------
# Non-standard geometry tests
# ---------------------------------------------------------------------------


def _custom_field(goal_x: float, goal_half_width: float):
    """Create a field stub with arbitrary goal line."""
    return SimpleNamespace(
        my_goal_line=np.array([(goal_x, goal_half_width), (goal_x, -goal_half_width)]),
        half_goal_width=goal_half_width,
    )


def test_goalkeep_custom_goal_line_changes_intercept_x(monkeypatch):
    """With a wider goal at x=-6.0, the keeper should target keeper_x = -6.0 + ROBOT_RADIUS."""
    keeper_x = -6.0 + ROBOT_RADIUS
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

    assert captured["target"].x == pytest.approx(keeper_x)


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
    keeper_x = -4.5 + ROBOT_RADIUS

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: Vector2D(x, 0.7))

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    # 0.7 < 0.8 half-width, so prediction is used directly
    assert captured["target"] == Vector2D(keeper_x, 0.7)


def test_goalkeep_wide_shot_clamps_to_post_limit(monkeypatch):
    """With goal_half_width=0.3, a prediction at y=0.4 is wide -> clamp to post_limit."""
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
    keeper_x = -4.5 + ROBOT_RADIUS
    post_limit = 0.3 - ROBOT_RADIUS

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: Vector2D(x, 0.4))

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    # abs(0.4) > 0.3 half-width -> clamp to post_limit
    assert captured["target"].x == pytest.approx(keeper_x)
    assert captured["target"].y == pytest.approx(post_limit)
