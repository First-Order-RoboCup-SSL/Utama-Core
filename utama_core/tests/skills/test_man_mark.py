from types import SimpleNamespace

import pytest

import utama_core.skills.src.man_mark as mm
from utama_core.entities.data.vector import Vector2D, Vector3D


def _make_game(friendly_robots, enemy_robots, ball_pos):
    return SimpleNamespace(
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=SimpleNamespace(p=Vector3D(ball_pos[0], ball_pos[1], 0.0)),
    )


def _capture_move(monkeypatch):
    captured = {}

    def fake(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
        captured["robot_id"] = robot_id
        captured["target_coords"] = target_coords
        captured["target_oren"] = target_oren
        return "sentinel-command"

    monkeypatch.setattr(mm, "move", fake)
    return captured


def test_man_mark_positions_between_ball_and_target(monkeypatch):
    """Marker should sit on the ball-to-opponent line, offset toward the
    opponent — not on top of either the ball or the opponent."""
    game = _make_game(
        friendly_robots={1: SimpleNamespace(p=Vector2D(0.0, 2.0), orientation=0.0)},
        enemy_robots={2: SimpleNamespace(p=Vector2D(2.0, 0.0))},
        ball_pos=(0.0, 0.0),
    )
    captured = _capture_move(monkeypatch)

    mm.man_mark(game, motion_controller=object(), robot_id=1, target_id=2)

    target = captured["target_coords"]
    # Perpendicular offset of magnitude 0.5 from the opponent's position,
    # along the ball->opponent line (2,0)->normalized (1,0), perpendicular (0,1).
    assert target.x == pytest.approx(2.0)
    assert target.y == pytest.approx(0.5)


def test_man_mark_calls_move_with_correct_robot_id(monkeypatch):
    game = _make_game(
        friendly_robots={3: SimpleNamespace(p=Vector2D(0.0, 0.0), orientation=0.0)},
        enemy_robots={4: SimpleNamespace(p=Vector2D(1.0, 1.0))},
        ball_pos=(0.0, 1.0),
    )
    captured = _capture_move(monkeypatch)

    mm.man_mark(game, motion_controller=object(), robot_id=3, target_id=4)

    assert captured["robot_id"] == 3


def test_man_mark_faces_the_ball(monkeypatch):
    game = _make_game(
        friendly_robots={1: SimpleNamespace(p=Vector2D(0.0, 0.0), orientation=0.0)},
        enemy_robots={2: SimpleNamespace(p=Vector2D(1.0, 0.0))},
        ball_pos=(0.0, 3.0),
    )
    captured = _capture_move(monkeypatch)

    mm.man_mark(game, motion_controller=object(), robot_id=1, target_id=2)

    # Robot is at origin, ball at (0, 3) -> facing angle should be pi/2 (straight up).
    assert captured["target_oren"] == pytest.approx(1.5707963267948966)


def test_man_mark_returns_a_command(monkeypatch):
    game = _make_game(
        friendly_robots={1: SimpleNamespace(p=Vector2D(0.0, 0.0), orientation=0.0)},
        enemy_robots={2: SimpleNamespace(p=Vector2D(1.0, 1.0))},
        ball_pos=(0.0, 0.0),
    )
    _capture_move(monkeypatch)

    result = mm.man_mark(game, motion_controller=object(), robot_id=1, target_id=2)

    assert result == "sentinel-command"
