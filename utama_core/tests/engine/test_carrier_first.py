"""`carrier_first`'s kicker at a still ball: a free-kick taker counts, a contested ball does not."""

from types import SimpleNamespace

from utama_core.strategy.pickers import _KICK_REACH_M, carrier_first


def _robot(x, has_ball=False):
    return SimpleNamespace(p=SimpleNamespace(x=x, y=0.0), has_ball=has_ball)


def _game(kicker_x, enemy_x):
    return SimpleNamespace(
        ball=SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0), v=SimpleNamespace(x=0.0, y=0.0)),
        friendly_robots={1: _robot(0.8), 3: _robot(kicker_x)},
        enemy_robots={0: _robot(enemy_x)},
    )


def test_a_kicker_waiting_at_a_still_ball_goes_first():
    assert carrier_first(_game(0.11, 0.6), frozenset({1, 3})) == [3, 1]


def test_a_contested_ball_has_no_waiting_kicker_whichever_robot_is_nearer():
    assert carrier_first(_game(0.111, 0.112), frozenset({1, 3})) == [1, 3]
    assert carrier_first(_game(0.113, 0.112), frozenset({1, 3})) == [1, 3]


def test_an_enemy_just_out_of_reach_leaves_the_kicker_first():
    assert carrier_first(_game(0.11, _KICK_REACH_M + 0.001), frozenset({1, 3})) == [3, 1]


def test_a_robot_with_the_ball_goes_first_even_when_contested():
    game = _game(0.11, 0.112)
    game.friendly_robots[3].has_ball = True
    assert carrier_first(game, frozenset({1, 3})) == [3, 1]
