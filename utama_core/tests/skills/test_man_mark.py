from types import SimpleNamespace

import pytest

import utama_core.skills.src.man_mark as mm
from utama_core.entities.data.vector import Vector2D, Vector3D


def _make_game(friendly_robots, enemy_robots, ball_pos, own_goal_x=4.5):
    return SimpleNamespace(
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=SimpleNamespace(p=Vector3D(ball_pos[0], ball_pos[1], 0.0)),
        field=SimpleNamespace(my_goal_line=((own_goal_x, -0.5), (own_goal_x, 0.5))),
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


def test_man_mark_stands_0_6m_from_the_opponent_toward_our_own_goal(monkeypatch):
    """Same spot `ShadowAndMarkTactic` marks from, whatever the ball does."""
    game = _make_game(
        friendly_robots={1: SimpleNamespace(p=Vector2D(0.0, 2.0), orientation=0.0)},
        enemy_robots={2: SimpleNamespace(p=Vector2D(2.0, 1.0))},
        ball_pos=(0.0, -2.0),
        own_goal_x=4.5,
    )
    captured = _capture_move(monkeypatch)

    mm.man_mark(game, motion_controller=object(), robot_id=1, target_id=2)

    assert captured["target_coords"].x == pytest.approx(2.6)
    assert captured["target_coords"].y == pytest.approx(1.0)

    game.field.my_goal_line = ((-4.5, -0.5), (-4.5, 0.5))
    mm.man_mark(game, motion_controller=object(), robot_id=1, target_id=2)
    assert captured["target_coords"].x == pytest.approx(1.4)


def test_two_robots_marking_each_other_stay_put(monkeypatch):
    """A `ShadowAndMarkTactic`-style marker (A, own goal at +x) and a `man_mark` robot (B, own
    goal at -x) marking each other. The old perpendicular offset moved B 0.5 m sideways every
    cycle and A followed, so the pair walked to the wall (docs/strategies.md, 2026-10-05)."""
    captured = _capture_move(monkeypatch)
    ball = (0.0, 0.0)
    a, b = Vector2D(3.6, 0.0), Vector2D(3.0, 0.0)
    for _ in range(60):
        # A stands 0.6 m from B toward A's goal (+x): ShadowAndMarkTactic's rule.
        a = Vector2D(b.x + 0.6, b.y)
        game_b = _make_game(
            friendly_robots={1: SimpleNamespace(p=b, orientation=0.0)},
            enemy_robots={2: SimpleNamespace(p=a)},
            ball_pos=ball,
            own_goal_x=-4.5,
        )
        mm.man_mark(game_b, motion_controller=object(), robot_id=1, target_id=2)
        b = captured["target_coords"]
    assert abs(b.x - 3.0) < 0.1
    assert abs(b.y) < 0.1


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
