from types import SimpleNamespace

from utama_core.entities.data.object import TeamType
from utama_core.strategy.kernel_strategy import _clear_press_plus_picker


def _final_third_game(carrier_id):
    distances = {TeamType.FRIENDLY: (carrier_id, 0.11), TeamType.ENEMY: (1, 0.21)}
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        my_team_is_right=True,
        field=SimpleNamespace(half_length=4.5),
        ball=SimpleNamespace(p=SimpleNamespace(to_2d=lambda: SimpleNamespace(x=-3.2))),
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
