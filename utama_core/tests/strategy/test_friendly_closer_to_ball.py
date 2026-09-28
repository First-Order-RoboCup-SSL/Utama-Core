from types import SimpleNamespace

from utama_core.entities.data.object import TeamType
from utama_core.strategy.kernel_strategy import _friendly_closer_to_ball


def _game(friendly_dist, enemy_dist, friendly_has_ball, enemy_has_ball=False):
    distances = {TeamType.FRIENDLY: (3, friendly_dist), TeamType.ENEMY: (2, enemy_dist)}
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        friendly_robots={3: SimpleNamespace(has_ball=friendly_has_ball)},
        enemy_robots={2: SimpleNamespace(has_ball=enemy_has_ball)},
    )


def test_a_ball_on_our_dribbler_is_ours_while_an_enemy_brushes_past():
    # counter_flow_vs_high_line_zone (2026-09-27): carrier 3 held the ball at
    # 0.119 m, an enemy pressed to 0.161 m, and the picker read "not ours",
    # handing the carrier to the press slot and resetting its give-and-go.
    assert _friendly_closer_to_ball(_game(0.119, 0.161, friendly_has_ball=True)) is True


def test_a_contested_ball_still_falls_back_to_distance():
    assert _friendly_closer_to_ball(_game(0.119, 0.161, friendly_has_ball=True, enemy_has_ball=True)) is False
