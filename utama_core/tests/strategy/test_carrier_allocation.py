from types import SimpleNamespace

from utama_core.entities.data.object import TeamType
from utama_core.strategy.kernel_strategy import _clear_press_plus_picker


def _final_third_game(carrier_id):
    distances = {TeamType.FRIENDLY: (carrier_id, 0.11), TeamType.ENEMY: (1, 0.21)}
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        my_team_is_right=True,
        field=SimpleNamespace(half_length=4.5),
        # A rolling ball: this case is about has_ball alone, not a kicker waiting at a still ball.
        ball=SimpleNamespace(p=SimpleNamespace(to_2d=lambda: SimpleNamespace(x=-3.2)), v=SimpleNamespace(x=1.0, y=0.0)),
        friendly_robots={rid: SimpleNamespace(has_ball=rid == carrier_id) for rid in range(1, 6)},
        enemy_robots={1: SimpleNamespace(has_ball=False)},
    )


def test_the_robot_holding_the_ball_goes_to_the_ball_side_slot():
    # clear_press_plus_vs_zone_fluid (2026-09-28): the final-third handoff gave
    # "overload" the two lowest ids and put carrier 4 in "block", which held the
    # ball still while the overload decoy waited on it — frozen for 38 s.
    partition = _clear_press_plus_picker(
        _final_third_game(carrier_id=4),
        frozenset({1, 2, 3, 4, 5}),
        {"attack": frozenset({3, 4, 5})},
        frozenset({"attack", "overload", "clear", "press", "block"}) - {"clear"},
    )
    assert 4 in partition["overload"]
    assert len(partition["overload"]) == 2


def _free_kick_game(kicker_dist, enemy_dist=0.8, ball_speed=0.0):
    def robot(dist):
        return SimpleNamespace(has_ball=False, p=SimpleNamespace(x=-dist, y=0.0))

    friendly = {rid: robot(1.0 + rid) for rid in range(1, 5)}
    friendly[5] = robot(kicker_dist)
    return SimpleNamespace(
        ball=SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0), v=SimpleNamespace(x=ball_speed, y=0.0)),
        friendly_robots=friendly,
        enemy_robots={1: robot(enemy_dist)},
    )


def test_a_free_kick_kicker_standing_at_the_ball_goes_first():
    # overload_press_vs_switch_of_play (2026-09-28): at NORMAL_START of our free
    # kick the kicker stood 0.11 m from the ball without dribbler contact, went to
    # "block"/"switch", and the overload decoy had to cross the field to it --
    # no kick for 10 s. Within kick reach of a still ball counts as holding it.
    from utama_core.strategy.kernel_strategy import _carrier_first

    assert _carrier_first(_free_kick_game(kicker_dist=0.12), frozenset(range(1, 6)))[0] == 5


def test_a_robot_short_of_kick_reach_or_beaten_to_the_ball_is_not_first():
    from utama_core.strategy.kernel_strategy import _carrier_first

    assert _carrier_first(_free_kick_game(kicker_dist=0.30), frozenset(range(1, 6)))[0] == 1
    assert _carrier_first(_free_kick_game(kicker_dist=0.12, enemy_dist=0.11), frozenset(range(1, 6)))[0] == 1
    assert _carrier_first(_free_kick_game(kicker_dist=0.12, ball_speed=1.0), frozenset(range(1, 6)))[0] == 1
