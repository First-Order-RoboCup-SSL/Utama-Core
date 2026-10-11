"""split_outlet_wall: the outlet slot allocation and the opponent direct-free-kick overrides.

Allocation tests are copied from test_split_outlet.py; override dispatch tests are adapted
from test_split_wall.py and run through the factory's own Strategy.
"""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.referee_override import RefereeOverride
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy import split_outlet_wall
from utama_core.strategy.split_outlet_wall import (
    _split_outlet_picker,
    build_split_outlet_wall_kernel_strategy,
)
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

ALL_SLOTS = frozenset({"attack", "outlet", "defense"})


def _robot(rid, x, y, is_friendly):
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _game(ball_xy, friendly, enemy, my_team_is_right=True, my_team_is_yellow=True):
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots={rid: _robot(rid, x, y, True) for rid, (x, y) in friendly.items()},
        enemy_robots={rid: _robot(rid, x, y, False) for rid, (x, y) in enemy.items()},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=my_team_is_right,
            field_dims=STANDARD_FIELD_DIMS,
            field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
        ),
    )


class _FakeMotion:
    def calculate(self, **kwargs):
        return Vector2D(0.0, 0.0), 0.0


def _split(n, *, ball_edge, available=ALL_SLOTS):
    """Partition for `n` free robots, with a friendly (edge) or enemy ball-edge game."""
    friendly = {rid: make_robot(rid, is_friendly=True, x=-1.0 - 0.5 * rid, y=0.0) for rid in range(1, n + 1)}
    if ball_edge:
        friendly[1] = make_robot(1, is_friendly=True, x=0.1, y=0.0)
        enemy = {1: make_robot(1, is_friendly=False, x=3.0, y=0.0)}
    else:
        enemy = {1: make_robot(1, is_friendly=False, x=0.2, y=0.0)}
    game = build_game(friendly, enemy, make_ball(0.0, 0.0), my_team_is_right=True)
    return _split_outlet_picker(game, frozenset(range(1, n + 1)), None, available)


def _sizes(partition):
    return {slot: len(robots) for slot, robots in partition.items()}


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, {"defense": 1}),
        (2, {"defense": 2}),
        (3, {"attack": 1, "outlet": 1, "defense": 1}),
        (4, {"attack": 1, "outlet": 1, "defense": 2}),
        (5, {"attack": 1, "outlet": 1, "defense": 3}),
    ],
)
def test_without_the_ball_edge_the_outlet_takes_one_robot_from_the_defence_from_three_up(n, expected):
    assert _sizes(_split(n, ball_edge=False)) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        # Exactly split_shape's split with the ball edge: no outlet slot at all.
        (1, {"attack": 1}),
        (2, {"attack": 2}),
        (3, {"attack": 3}),
        (4, {"attack": 3, "defense": 1}),
        (5, {"attack": 4, "defense": 1}),
    ],
)
def test_with_the_ball_edge_there_is_no_outlet_slot(n, expected):
    partition = _split(n, ball_edge=True)
    assert _sizes(partition) == expected
    assert "outlet" not in partition


def test_partition_covers_every_free_robot_exactly_once():
    for n in range(1, 6):
        for edge in (True, False):
            partition = _split(n, ball_edge=edge)
            union = frozenset().union(*partition.values())
            assert union == frozenset(range(1, n + 1))
            assert sum(len(v) for v in partition.values()) == n


def test_an_unavailable_outlet_hands_its_robot_to_the_defence():
    partition = _split(5, ball_edge=False, available=frozenset({"attack", "defense"}))
    assert _sizes(partition) == {"attack": 1, "defense": 4}
    assert "outlet" not in partition


def test_no_zero_robot_entries_are_emitted():
    partition = _split(2, ball_edge=False)
    assert all(len(v) > 0 for v in partition.values())
    assert "attack" not in partition


# ---- factory: both DIRECT_FREE overrides are wired ------------------------


def _strategy():
    # The factory's Strategy is built from a bare stand-in motion controller, as in
    # tests/engine/test_kernel_strategy.py.
    return build_split_outlet_wall_kernel_strategy((1, 2, 3, 4, 5))(object())


def test_factory_strategy_carries_both_direct_free_overrides():
    overrides = _strategy().referee_overrides
    assert set(overrides) == {RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.DIRECT_FREE_BLUE}


def test_factory_partitioner_is_the_outlet_picker(monkeypatch):
    captured = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(split_outlet_wall, "KernelSchedulerStrategy", _Spy)
    build_split_outlet_wall_kernel_strategy((1, 2, 3))(object())
    assert captured["partitioner"] is split_outlet_wall._split_outlet_picker
    assert set(captured["tactics"]) == {"attack", "outlet", "defense"}


def test_our_own_direct_free_kick_runs_the_built_in_step_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(split_outlet_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0), 3: (-2.5, -1.0)},
        enemy={0: (3.0, 0.0)},
        my_team_is_yellow=True,
    )
    ours = _strategy().referee_overrides[RefereeCommand.DIRECT_FREE_YELLOW](game, _FakeMotion())
    built_in = RefereeOverride().tick(game, _FakeMotion(), RefereeCommand.DIRECT_FREE_YELLOW)

    assert calls == []
    assert ours == built_in


def test_opponent_direct_free_kick_gets_the_wall(monkeypatch):
    calls = []
    monkeypatch.setattr(split_outlet_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    # We are blue, so the opponent's (yellow) kick is DIRECT_FREE_YELLOW and must be walled.
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0)},
        enemy={0: (0.1, 0.0)},
        my_team_is_yellow=False,
    )
    _strategy().referee_overrides[RefereeCommand.DIRECT_FREE_YELLOW](game, _FakeMotion())
    assert calls == [1]


def test_blue_key_is_the_opponents_kick_when_we_are_yellow(monkeypatch):
    calls = []
    monkeypatch.setattr(split_outlet_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0)},
        enemy={0: (0.1, 0.0)},
        my_team_is_yellow=True,
    )
    _strategy().referee_overrides[RefereeCommand.DIRECT_FREE_BLUE](game, _FakeMotion())
    assert calls == [1]


def test_wall_targets_are_the_split_wall_geometry_for_the_outlet_robot_too():
    # With the outlet robot among the outfield robots, it is one of the wall/markers at the
    # opponent's kick (the override controls every outfield robot), not held by the picker.
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (0.5, -0.2), 3: (-2.0, 1.0)},
        enemy={0: (0.1, 0.0)},
        my_team_is_yellow=False,
    )
    targets = split_outlet_wall.opponent_free_kick_targets(game, (1, 2, 3))
    assert set(targets) == {1, 2, 3}
    assert targets[1][0].x == pytest.approx(0.9) and targets[1][0].y == pytest.approx(0.1)
    assert targets[2][0].x == pytest.approx(0.9) and targets[2][0].y == pytest.approx(-0.1)
    assert isinstance(targets[3][0], Vector2D)
