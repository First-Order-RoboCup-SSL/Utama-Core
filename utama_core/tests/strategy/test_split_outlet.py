"""split_outlet: the outlet holding point (both attack directions) and the slot allocation."""

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.shared.pass_and_score_geometry import enemy_goal_line
from utama_core.strategy.split_outlet import _split_outlet_picker
from utama_core.tactics.outlet import holding_points
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

ALL_SLOTS = frozenset({"attack", "outlet", "defense"})


def _game(ball_xy=(0.0, 0.0), my_team_is_right=True, friendly=None, enemy=None):
    friendly = friendly if friendly is not None else {1: make_robot(1, is_friendly=True, x=-3.0, y=0.0)}
    enemy = enemy if enemy is not None else {}
    return build_game(
        friendly,
        enemy,
        make_ball(*ball_xy),
        my_team_is_right=my_team_is_right,
    )


@pytest.mark.parametrize("mine_right", [True, False])
def test_holding_point_is_a_quarter_of_the_way_to_the_enemy_goal(mine_right):
    game = _game(ball_xy=(0.0, 0.0), my_team_is_right=mine_right)
    goal_x, _, _ = enemy_goal_line(game)
    assert goal_x == (-4.5 if mine_right else 4.5)

    (point,) = holding_points(game, 1)
    assert point.x == pytest.approx(0.25 * goal_x)
    # The point is on the enemy side of halfway: the same sign as the enemy goal.
    assert point.x * goal_x > 0


@pytest.mark.parametrize("mine_right", [True, False])
def test_holding_point_sits_on_the_side_away_from_the_ball(mine_right):
    game = _game(ball_xy=(0.0, 2.0), my_team_is_right=mine_right)
    (point,) = holding_points(game, 1)
    assert point.y == pytest.approx(-0.6 * 2.0)

    game = _game(ball_xy=(0.0, -1.0), my_team_is_right=mine_right)
    (point,) = holding_points(game, 1)
    assert point.y == pytest.approx(0.6)


def test_holding_point_y_is_clamped_to_the_touchline_margin():
    # Ball at the far touchline on the left: -0.6 * (-5) = 3.0 would sit 0.5 m outside
    # the margin (half_width 3.0 - 0.5 = 2.5), so the point stops at 2.5.
    game = _game(ball_xy=(0.0, -5.0))
    (point,) = holding_points(game, 1)
    assert point.y == pytest.approx(game.field.half_width - 0.5)


def test_multiple_holding_points_are_spread_along_y_and_stay_inside_the_margin():
    game = _game(ball_xy=(0.0, 0.0))
    points = holding_points(game, 2)
    assert len(points) == 2
    assert points[0].y == pytest.approx(-0.75)
    assert points[1].y == pytest.approx(0.75)
    assert all(abs(p.y) <= game.field.half_width - 0.5 for p in points)


def test_holding_point_is_never_inside_the_enemy_defense_area():
    # Ball deep in the enemy half should not drag the point into the enemy box.
    game = _game(ball_xy=(3.5, 0.0), my_team_is_right=True)
    (point,) = holding_points(game, 1)
    defense_area = game.field.enemy_defense_area
    front_x = float(defense_area[1][0])
    assert not (point.x < front_x and abs(point.y) < abs(float(defense_area[0][1])))
    assert isinstance(point, Vector2D)


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
