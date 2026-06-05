from types import SimpleNamespace

import numpy as np
import pytest

import utama_core.skills.src.defend_parameter as dp
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D

EDGE_OFFSET = BALL_RADIUS + ROBOT_RADIUS

# ---------------------------------------------------------------------------
# Standard-field stubs (9x6 field)
# goal_x = +/-4.5, goal_half_width = 0.5
# defense front at +/-3.5, defense_half_width = 1.0
# defender_x (left) = -3.5 + ROBOT_RADIUS, defender_x (right) = 3.5 - ROBOT_RADIUS
# keeper_x (left) = -4.5 + ROBOT_RADIUS, keeper_x (right) = 4.5 - ROBOT_RADIUS
# post_limit = 0.5 - ROBOT_RADIUS
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

# Defender stands one ROBOT_RADIUS in front of the defense area
_LEFT_DEFENDER_X = -3.5 + ROBOT_RADIUS
_RIGHT_DEFENDER_X = 3.5 - ROBOT_RADIUS

# Keeper-line references
_LEFT_KEEPER_X = -4.5 + ROBOT_RADIUS
_RIGHT_KEEPER_X = 4.5 - ROBOT_RADIUS
_STD_POST_LIMIT = 0.5 - ROBOT_RADIUS


def _std_field(my_team_is_right: bool):
    if my_team_is_right:
        return SimpleNamespace(
            my_goal_line=_STD_RIGHT_GOAL_LINE,
            my_defense_area=_STD_RIGHT_DEFENSE_AREA,
            half_goal_width=0.5,
        )
    return SimpleNamespace(
        my_goal_line=_STD_LEFT_GOAL_LINE,
        my_defense_area=_STD_LEFT_DEFENSE_AREA,
        half_goal_width=0.5,
    )


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
        half_goal_width=goal_half_width,
    )


def _make_game(team_is_right, field, friendly_robots, ball_pos):
    return SimpleNamespace(
        my_team_is_right=team_is_right,
        field=field,
        friendly_robots=friendly_robots,
        ball=SimpleNamespace(
            p=Vector3D(ball_pos[0], ball_pos[1], 0.0),
            v=Vector3D(0.0, 0.0, 0.0),
        ),
    )


def _capture_go_to_point(monkeypatch):
    captured = {}

    def fake(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        captured["dribbling"] = dribbling
        captured["robot_id"] = robot_id
        return "sentinel-command"

    monkeypatch.setattr(dp, "go_to_point", fake)
    return captured


# ---------------------------------------------------------------------------
# Core geometry: defender on ball-to-post line using keeper-x reference
# ---------------------------------------------------------------------------


def test_defender_on_ball_to_post_line_left_team(monkeypatch):
    """Left team, robot 1 covers top post. Ball at origin."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # goal_point = (keeper_x, post_limit + EDGE_OFFSET)
    # t = (defender_x - 0) / (keeper_x - 0)
    t = _LEFT_DEFENDER_X / _LEFT_KEEPER_X
    expected_y = t * (_STD_POST_LIMIT + EDGE_OFFSET)
    assert captured["target"].x == pytest.approx(_LEFT_DEFENDER_X)
    assert captured["target"].y == pytest.approx(expected_y)
    assert captured["dribbling"] is True


def test_defender_on_ball_to_post_line_right_team(monkeypatch):
    """Right team, robot 1 covers top post. Ball at origin."""
    game = _make_game(
        team_is_right=True,
        field=_std_field(True),
        friendly_robots={1: SimpleNamespace(p=Vector2D(3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # goal_point = (keeper_x, post_limit + EDGE_OFFSET)
    t = _RIGHT_DEFENDER_X / _RIGHT_KEEPER_X
    expected_y = t * (_STD_POST_LIMIT + EDGE_OFFSET)
    assert captured["target"].x == pytest.approx(_RIGHT_DEFENDER_X)
    assert captured["target"].y == pytest.approx(expected_y)


def test_defender_robot2_covers_bottom_post(monkeypatch):
    """Robot 2 defaults to the bottom post (-post_limit)."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={2: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=2)

    t = _LEFT_DEFENDER_X / _LEFT_KEEPER_X
    expected_y = t * (-_STD_POST_LIMIT - EDGE_OFFSET)
    assert captured["target"].x == pytest.approx(_LEFT_DEFENDER_X)
    assert captured["target"].y == pytest.approx(expected_y)


# ---------------------------------------------------------------------------
# goal_frame_y=0.0 targets centre of goal (explicit override still works)
# ---------------------------------------------------------------------------


def test_goal_frame_y_zero_targets_centre(monkeypatch):
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.5),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.0)

    # goal_point = (keeper_x, EDGE_OFFSET)
    t = _LEFT_DEFENDER_X / _LEFT_KEEPER_X
    expected_y = 0.5 + t * (EDGE_OFFSET - 0.5)
    assert captured["target"].x == pytest.approx(_LEFT_DEFENDER_X)
    assert captured["target"].y == pytest.approx(expected_y)
    assert captured["dribbling"] is True


# ---------------------------------------------------------------------------
# Multi-robot: both defenders dynamically track (no static positions)
# ---------------------------------------------------------------------------


def test_three_robots_both_defenders_dynamic(monkeypatch):
    """With 3+ robots, both defenders should track the ball-to-post line."""
    field = _std_field(False)
    robots = {
        0: SimpleNamespace(p=Vector2D(-4.5, 0.0)),
        1: SimpleNamespace(p=Vector2D(-3.0, 0.3)),
        2: SimpleNamespace(p=Vector2D(-3.0, -0.3)),
    }
    captured_calls = []

    def fake(game, motion_controller, robot_id, target, dribbling=False):
        captured_calls.append({"robot_id": robot_id, "target": target})
        return "sentinel-command"

    monkeypatch.setattr(dp, "go_to_point", fake)

    game1 = _make_game(False, field, robots, (0.0, 0.0))
    dp.defend_parameter(game1, motion_controller=object(), robot_id=1)

    game2 = _make_game(False, field, robots, (0.0, 0.0))
    dp.defend_parameter(game2, motion_controller=object(), robot_id=2)

    assert len(captured_calls) == 2
    # Robot 1 covers top post -> positive y
    assert captured_calls[0]["target"].y > 0
    # Robot 2 covers bottom post -> negative y
    assert captured_calls[1]["target"].y < 0
    # Both just in front of defense area
    assert captured_calls[0]["target"].x == pytest.approx(_LEFT_DEFENDER_X)
    assert captured_calls[1]["target"].x == pytest.approx(_LEFT_DEFENDER_X)


# ---------------------------------------------------------------------------
# Y clamped to defense_half_width
# ---------------------------------------------------------------------------


def test_target_y_clamped_to_defense_half_width(monkeypatch):
    """When ball-to-post line exits defense area vertically, y is clamped."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 3.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    assert captured["target"].y == pytest.approx(1.0)
    assert captured["target"].x == pytest.approx(_LEFT_DEFENDER_X)


# ---------------------------------------------------------------------------
# Custom geometry
# ---------------------------------------------------------------------------


def test_custom_geometry_uses_field_values(monkeypatch):
    """Defender positioning adapts to non-standard field dimensions."""
    field = _custom_field(-6.0, 0.8, -4.5, 1.5)
    keeper_x = -6.0 + ROBOT_RADIUS
    post_limit = 0.8 - ROBOT_RADIUS
    defender_x = -4.5 + ROBOT_RADIUS  # left team
    game = _make_game(
        team_is_right=False,
        field=field,
        friendly_robots={1: SimpleNamespace(p=Vector2D(-4.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # goal_point = (keeper_x, post_limit + EDGE_OFFSET)
    t = defender_x / keeper_x
    expected_y = t * (post_limit + EDGE_OFFSET)
    assert captured["target"].x == pytest.approx(defender_x)
    assert captured["target"].y == pytest.approx(expected_y)


def test_custom_geometry_default_goal_frame_y(monkeypatch):
    """Default goal_frame_y uses post_limit from field, not hardcoded 0.5."""
    field = _custom_field(-4.5, 0.8, -3.5, 1.0)
    post_limit = 0.8 - ROBOT_RADIUS
    game = _make_game(
        team_is_right=False,
        field=field,
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # goal_frame_y = post_limit (not 0.5 or 0.8)
    # goal_point = (keeper_x, post_limit + EDGE_OFFSET)
    keeper_x = -4.5 + ROBOT_RADIUS
    t = _LEFT_DEFENDER_X / keeper_x
    expected_y = t * (post_limit + EDGE_OFFSET)
    assert captured["target"].y == pytest.approx(expected_y)


def test_custom_geometry_y_clamped_to_custom_defense_half_width(monkeypatch):
    """Y clamping uses custom defense_half_width, not hardcoded 1.0."""
    field = _custom_field(-4.5, 0.5, -3.5, 0.3)
    game = _make_game(
        team_is_right=False,
        field=field,
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # Unclamped y > 0.3 -> clamped
    assert captured["target"].y == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Radius-aware target geometry
# ---------------------------------------------------------------------------


def test_shadow_target_uses_keeper_x_not_goal_x(monkeypatch):
    """The shadow target x-coordinate is keeper_x, not the raw goal line."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={1: SimpleNamespace(p=Vector2D(-3.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1, goal_frame_y=0.3)

    # If using keeper_x (-4.41) vs goal_x (-4.5), the t parameter differs.
    # Verify by checking y matches the keeper_x-based formula:
    t = _LEFT_DEFENDER_X / _LEFT_KEEPER_X
    expected_y = t * (0.3 + EDGE_OFFSET)
    assert captured["target"].y == pytest.approx(expected_y)


# ---------------------------------------------------------------------------
# Single-defender dynamic side choice (2-robot case)
# ---------------------------------------------------------------------------


def test_single_defender_picks_side_for_off_centre_ball(monkeypatch):
    """When ball is clearly off-centre, the defender should pick the correct side."""
    # Ball far to the positive-y side: defender should cover +post_limit
    # so that the keeper doesn't have to travel far to cover the near post
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.41, 0.0)),  # goalkeeper
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),  # defender
        },
        ball_pos=(0.0, 2.0),
    )
    captured = _capture_go_to_point(monkeypatch)
    monkeypatch.setattr(dp, "predict_ball_pos_at_x", lambda game, x: None)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    # Defender should be shifted toward the ball side (positive y)
    assert captured["target"].y > 0


def test_single_defender_picks_opposite_side_for_negative_ball(monkeypatch):
    """Mirror: ball far negative-y, defender should cover -post_limit."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.41, 0.0)),
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
        },
        ball_pos=(0.0, -2.0),
    )
    captured = _capture_go_to_point(monkeypatch)
    monkeypatch.setattr(dp, "predict_ball_pos_at_x", lambda game, x: None)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)

    assert captured["target"].y < 0


def test_single_defender_centred_ball_uses_keeper_ref_tiebreak(monkeypatch):
    """With a centred ball, side choice considers keeper reference position."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.41, 0.2)),  # keeper slightly positive
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
        },
        ball_pos=(0.0, 0.0),
    )
    captured_a = _capture_go_to_point(monkeypatch)
    monkeypatch.setattr(dp, "predict_ball_pos_at_x", lambda game, x: None)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)
    y_a = captured_a["target"].y

    # Now mirror: keeper slightly negative
    game2 = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.41, -0.2)),
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
        },
        ball_pos=(0.0, 0.0),
    )
    captured_b = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game2, motion_controller=object(), robot_id=1)
    y_b = captured_b["target"].y

    # The two should mirror: keeper-positive -> defender-negative and vice versa
    assert y_a * y_b < 0  # opposite signs


def test_single_defender_uses_predicted_intercept_for_keeper_ref(monkeypatch):
    """When a ball prediction is available, it is used as the keeper reference."""
    game = _make_game(
        team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.41, 0.0)),  # keeper at centre
            1: SimpleNamespace(p=Vector2D(-3.0, 0.0)),
        },
        ball_pos=(0.0, 0.0),
    )
    # Predict ball heading to positive y
    monkeypatch.setattr(
        dp,
        "predict_ball_pos_at_x",
        lambda game, x: Vector2D(x, 0.3),
    )
    captured = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)
    y_with_positive_prediction = captured["target"].y

    # Predict ball heading to negative y
    monkeypatch.setattr(
        dp,
        "predict_ball_pos_at_x",
        lambda game, x: Vector2D(x, -0.3),
    )
    captured2 = _capture_go_to_point(monkeypatch)

    dp.defend_parameter(game, motion_controller=object(), robot_id=1)
    y_with_negative_prediction = captured2["target"].y

    # Different predictions should produce different (mirrored) side choices
    assert y_with_positive_prediction * y_with_negative_prediction < 0
