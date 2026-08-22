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


def _defense_area(goal_x: float, depth: float, width: float) -> np.ndarray:
    """Same 4-corner shape `FieldDimensions.{left,right}_defense_area` returns
    (see `field_params.py`) — front edge `2*depth` in from the goal line,
    toward center; y in `[-width, width]`. `goalkeep`'s new box-retrieval
    branch reads this via `in_own_defense_area`/`own_defense_area_exit_point`,
    so the stub needs a real one now, not just the goal line."""
    front_x = goal_x + 2 * depth if goal_x < 0 else goal_x - 2 * depth
    return np.array([(goal_x, width), (front_x, width), (front_x, -width), (goal_x, -width)])


def _std_field(my_team_is_right: bool):
    goal_line = _STD_RIGHT_GOAL_LINE if my_team_is_right else _STD_LEFT_GOAL_LINE
    goal_x = 4.5 if my_team_is_right else -4.5
    return SimpleNamespace(
        my_goal_line=goal_line,
        half_goal_width=0.5,
        my_defense_area=_defense_area(goal_x, depth=0.5, width=1.0),
    )


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


def test_goalkeep_single_keeper_no_prediction_tracks_ball_y(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-4.2, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-1.0, 0.2, 0.0),
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
    assert captured["target"] == Vector2D(_LEFT_KEEPER_X, 0.2)


def test_goalkeep_single_keeper_no_prediction_clamps_to_upper_post(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_custom_field(-1.5, 0.4),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-1.35, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-0.679, 0.623, 0.0),
            v=Vector3D(0.078, -2.35, 0.0),
        ),
    )
    captured = {}
    keeper_x = -1.5 + ROBOT_RADIUS
    post_limit = 0.4 - ROBOT_RADIUS

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert captured["target"].x == pytest.approx(keeper_x)
    assert captured["target"].y == pytest.approx(post_limit)


def test_goalkeep_single_keeper_no_prediction_clamps_to_lower_post(monkeypatch):
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_custom_field(-1.5, 0.4),
        friendly_robots={
            0: SimpleNamespace(p=Vector2D(-1.35, 0.0)),
        },
        ball=SimpleNamespace(
            p=Vector3D(-0.626, -0.495, 0.0),
            v=Vector3D(0.002, 0.001, 0.0),
        ),
    )
    captured = {}
    keeper_x = -1.5 + ROBOT_RADIUS
    post_limit = 0.4 - ROBOT_RADIUS

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-command"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert captured["target"].x == pytest.approx(keeper_x)
    assert captured["target"].y == pytest.approx(-post_limit)


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
    """Create a field stub with arbitrary goal line.

    `depth`/`width` scaled down from the standard field's (0.5/1.0) in
    proportion to `goal_half_width`, the same ratio `GREAT_EXHIBITION_FIELD_DIMS`
    uses (see `field_params.py`) — these tests use a small custom field
    (goal_x=-1.5), and a standard-scale box would swallow ball positions
    these tests place well outside any box on their own field's scale.
    """
    depth = 0.5 * (goal_half_width / 0.5) * 0.5
    width = 1.0 * (goal_half_width / 0.5) * 0.5
    return SimpleNamespace(
        my_goal_line=np.array([(goal_x, goal_half_width), (goal_x, -goal_half_width)]),
        half_goal_width=goal_half_width,
        my_defense_area=_defense_area(goal_x, depth=depth, width=width),
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


# ---------------------------------------------------------------------------
# Box-retrieval / clearing (ball at rest inside our own defense area)
# ---------------------------------------------------------------------------


def test_goalkeep_drives_to_ball_at_rest_in_own_box(monkeypatch):
    """A near-stationary ball deep in the box (front edge at x=-3.5, well off
    the goal line at x=-4.5) must be driven straight to, not treated as a
    goal-line shot to cover — see `_ball_needs_retrieval`."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
        friendly_robots={0: SimpleNamespace(p=Vector2D(-4.2, 0.0), has_ball=False)},
        ball=SimpleNamespace(p=Vector3D(-4.0, 0.3, 0.0), v=Vector3D(0.05, 0.0, 0.0)),
    )
    captured = {}

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        captured["dribbling"] = dribbling
        return "sentinel-retrieve"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-retrieve"
    assert captured["target"] == Vector2D(-4.0, 0.3)
    assert captured["dribbling"] is True


def test_goalkeep_ignores_fast_ball_in_box_treats_as_shot(monkeypatch):
    """A fast-moving ball inside the box (a live shot, not a dead ball) must
    fall through to ordinary goal-line targeting, not the retrieval branch."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
        friendly_robots={0: SimpleNamespace(p=Vector2D(-4.2, 0.0), has_ball=False)},
        ball=SimpleNamespace(p=Vector3D(-4.0, 0.3, 0.0), v=Vector3D(2.0, 0.0, 0.0)),
    )
    captured = {}

    monkeypatch.setattr(gk, "predict_ball_pos_at_x", lambda game, x: None)

    def fake_go_to_point(game, motion_controller, robot_id, target, dribbling=False):
        captured["target"] = target
        return "sentinel-shot"

    monkeypatch.setattr(gk, "go_to_point", fake_go_to_point)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-shot"
    # Ordinary goal-line branch: x pinned to keeper_x, not the ball's own x.
    assert captured["target"].x == pytest.approx(_LEFT_KEEPER_X)


def test_goalkeep_dribbles_retrieved_ball_toward_box_exit(monkeypatch):
    """Once the keeper has picked up a ball inside the box, it must dribble
    toward the box exit point (`own_defense_area_exit_point`), not sit still
    or fall back to a goal-line target that would drag the ball toward our
    own goal — see `_ball_needs_clearing`."""
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
        friendly_robots={0: SimpleNamespace(p=Vector2D(-4.0, 0.3), orientation=0.0, has_ball=True)},
        ball=SimpleNamespace(p=Vector3D(-4.0, 0.3, 0.0), v=Vector3D(0.0, 0.0, 0.0)),
    )
    captured = {}

    def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
        captured["target_coords"] = target_coords
        captured["dribbling"] = dribbling
        return "sentinel-dribble-out"

    monkeypatch.setattr(gk, "move", fake_move)

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-dribble-out"
    # own_defense_area_exit_point sits just outside the box's front edge
    # (x=-3.5 for the standard field stub) by a robot-diameter margin, at
    # x≈-3.27; keeper is still ~0.7m short of it, so it must dribble there,
    # not kick yet.
    assert captured["target_coords"].x == pytest.approx(-3.27, abs=0.05)
    assert captured["dribbling"] is True


def test_goalkeep_kicks_once_at_box_exit_and_oriented(monkeypatch):
    """At the box exit point and facing upfield, the keeper must kick rather
    than keep dribbling — completing the clearance."""
    exit_x = -3.27  # own_defense_area_exit_point's x for this stub — see the dribble test above
    # `_ball_needs_clearing` latches on `has_ball` once retrieval starts and
    # does NOT re-check box position (see its docstring — a dribbled ball
    # tracks the keeper, so by arrival it has already crossed just outside
    # the box). Keeper (and the ball, glued to it) sit at the exit point.
    game = SimpleNamespace(
        my_team_is_right=False,
        field=_std_field(False),
        # Facing +x (upfield, away from our own goal at -4.5) — already
        # oriented toward the clearance target this close to the exit point.
        friendly_robots={0: SimpleNamespace(p=Vector2D(exit_x, 0.0), orientation=0.0, has_ball=True)},
        ball=SimpleNamespace(p=Vector3D(exit_x, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0)),
    )

    monkeypatch.setattr(gk, "kick", lambda: "sentinel-kick")

    result = gk.goalkeep(game, motion_controller=object(), robot_id=0)

    assert result == "sentinel-kick"
