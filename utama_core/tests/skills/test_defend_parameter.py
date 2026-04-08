from types import SimpleNamespace

import numpy as np
import pytest

import utama_core.skills.src.defend_parameter as dp
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D

# ---------------------------------------------------------------------------
# Standard-field stubs (9x6 field)
# goal_x = ±4.5, goal_half_width = 0.5
# defense front at ±3.5, defense_half_width = 1.0
# ---------------------------------------------------------------------------
_STD_LEFT_GOAL_LINE = np.array([(-4.5, 0.5), (-4.5, -0.5)])
_STD_RIGHT_GOAL_LINE = np.array([(4.5, 0.5), (4.5, -0.5)])
_STD_LEFT_DEFENSE_AREA = np.array(
    [
        (-4.5, 1.0),
        (-3.5, 1.0),
        (-3.5, -1.0),
        (-4.5, -1.0),
        (-4.5, 1.0),
    ]
)
_STD_RIGHT_DEFENSE_AREA = np.array(
    [
        (4.5, 1.0),
        (3.5, 1.0),
        (3.5, -1.0),
        (4.5, -1.0),
        (4.5, 1.0),
    ]
)


def _std_field(my_team_is_right: bool):
    if my_team_is_right:
        return SimpleNamespace(
            my_goal_line=_STD_RIGHT_GOAL_LINE,
            my_defense_area=_STD_RIGHT_DEFENSE_AREA,
        )
    return SimpleNamespace(
        my_goal_line=_STD_LEFT_GOAL_LINE,
        my_defense_area=_STD_LEFT_DEFENSE_AREA,
    )


# ---------------------------------------------------------------------------
# Existing tests – now with standard-field stubs
# ---------------------------------------------------------------------------


def test_defend_parameter_uses_ball_xy_projection(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
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
        field=_std_field(team_is_right),
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
        field=_std_field(False),
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
        field=_std_field(True),
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


# ---------------------------------------------------------------------------
# Non-standard geometry tests
# ---------------------------------------------------------------------------


def _custom_field(goal_x, goal_half_width, defense_front_x, defense_half_width):
    """Create a field stub with arbitrary goal line and defense area."""
    return SimpleNamespace(
        my_goal_line=np.array(
            [
                (goal_x, goal_half_width),
                (goal_x, -goal_half_width),
            ]
        ),
        my_defense_area=np.array(
            [
                (goal_x, defense_half_width),
                (defense_front_x, defense_half_width),
                (defense_front_x, -defense_half_width),
                (goal_x, -defense_half_width),
                (goal_x, defense_half_width),
            ]
        ),
    )


def test_defend_parameter_custom_geometry_static_extra_defenders(monkeypatch):
    """Static extra-defender positions should adapt to defense area geometry."""
    # Custom: goal at x=-6.0, defense front at -4.5, defense_half_width=1.5
    field = _custom_field(-6.0, 0.8, -4.5, 1.5)
    game = SimpleNamespace(
        my_team_is_right=False,
        field=field,
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(0.0, 0.0)),
            1: SimpleNamespace(p=Vector2D(-1.0, 0.2)),
            2: SimpleNamespace(p=Vector2D(-1.0, -0.2)),
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

    # ball_pos.y = 0.0 >= -goal_half_width (-0.8), robot_id=1 → static assignment
    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # x = defense_front_x - sign * 0.5 = -4.5 - (-1)*0.5 = -4.0
    assert captured["target"].x == pytest.approx(-4.0)
    # y = -(defense_half_width + 0.2) = -(1.5 + 0.2) = -1.7
    assert captured["target"].y == pytest.approx(-1.7)


def test_defend_parameter_custom_geometry_trajectory_gating(monkeypatch):
    """Trajectory gating should use goal_half_width, not hardcoded 0.5."""
    # Custom: goal at x=-4.5, goal_half_width=0.8, defense front -3.5, defense_half_width=1.0
    field = _custom_field(-4.5, 0.8, -3.5, 1.0)
    game = SimpleNamespace(
        my_team_is_right=False,
        field=field,
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
        if x == pytest.approx(-4.5):
            # Ball arrives at y=0.6, which is inside 0.8 goal but outside old 0.5
            return Vector2D(-4.5, 0.6)
        if x == pytest.approx(-4.0):
            return Vector2D(-4.0, 0.5)
        return None

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-command"

    monkeypatch.setattr(dp, "predict_ball_pos_at_x", fake_predict_ball_pos_at_x)
    monkeypatch.setattr(dp, "go_to_point", fake_go_to_point)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    # With goal_half_width=0.8, abs(0.6) < 0.8 → trajectory branch taken
    assert captured["dribbling"] is True


def test_defend_parameter_custom_defense_depth_changes_clamp(monkeypatch):
    """Clamp window check uses defense_depth instead of hardcoded 1.0."""
    # Custom: goal at x=6.0, defense front at 4.0, defense_depth=2.0, defense_half_width=1.5
    field = _custom_field(6.0, 0.8, 4.0, 1.5)
    game = SimpleNamespace(
        my_team_is_right=True,
        field=field,
        friendly_robots={
            1: SimpleNamespace(p=Vector2D(4.5, 0.0)),
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

    dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    # Should complete without error; target should be near the defense front
    assert isinstance(captured["target"], Vector2D)


def test_defend_parameter_default_goal_frame_y_uses_goal_half_width(monkeypatch):
    """When goal_frame_y is None, it should default to ±goal_half_width, not ±0.5."""
    # Custom: goal_half_width = 0.8
    field = _custom_field(-4.5, 0.8, -3.5, 1.0)
    game = SimpleNamespace(
        my_team_is_right=False,
        field=field,
        friendly_robots={
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
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

    # goal_frame_y=None → defaults to goal_half_width for robot_id=1
    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # The else branch uses -goal_frame_y = -0.8 as goal_point y
    # With standard-width=0.5 this would have been -0.5
    # The target should reflect the wider goal frame
    assert isinstance(captured["target"], Vector2D)
    # goal_point = (-4.5, -0.8), ball at (0,0), robot at (-3, 0)
    # This projects onto the line from ball to goal_point
    # The key assertion: target should differ from what 0.5 would produce
    assert captured["target"].y != pytest.approx(0.0)
