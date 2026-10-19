from types import SimpleNamespace

import pytest

from utama_core.entities.data.object import TeamType
from utama_core.strategy.kernel_strategy import (
    friendly_closer_to_ball,
    reset_closer_to_ball_state,
)


@pytest.fixture(autouse=True)
def _reset_state():
    reset_closer_to_ball_state()
    yield
    reset_closer_to_ball_state()


def _game(friendly_dist, enemy_dist, friendly_has_ball=False, enemy_has_ball=False):
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
    assert friendly_closer_to_ball(_game(0.119, 0.161, friendly_has_ball=True)) is True


def test_a_contested_ball_still_falls_back_to_distance():
    # Enemy clearly nearer (gap 0.381 m, well outside the 5 cm margin).
    assert friendly_closer_to_ball(_game(0.5, 0.119, friendly_has_ball=False)) is False
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.119, 0.5, friendly_has_ball=False)) is True


def test_a_near_tie_inside_the_margin_is_undecided_not_lost():
    # Both directions inside the 5 cm margin: neither side is clearly ahead, so
    # the sticky pickers (`edge is False` to give up possession) must keep their
    # previous state. This used to return False and drop the ball on noise.
    assert friendly_closer_to_ball(_game(0.119, 0.161, friendly_has_ball=True, enemy_has_ball=True)) is None
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.161, 0.119, friendly_has_ball=False)) is None
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.30, 0.30, friendly_has_ball=False)) is None


def test_margin_boundary_is_exclusive_on_both_sides():
    # Just inside the margin is undecided; just past it decides.
    assert friendly_closer_to_ball(_game(0.10, 0.1499, friendly_has_ball=False)) is None
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.1499, 0.10, friendly_has_ball=False)) is None
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.10, 0.1501, friendly_has_ball=False)) is True
    reset_closer_to_ball_state()
    assert friendly_closer_to_ball(_game(0.1501, 0.10, friendly_has_ball=False)) is False


def test_possession_hysteresis_holds_through_transient_distance_flicker():
    # Friendly acquires possession clearly
    assert friendly_closer_to_ball(_game(0.1, 0.4)) is True

    # Enemy gets closer for 14 ticks (< 15 ticks / ~0.25 s) within break margin:
    # possession must hold True without flickering to False.
    for _ in range(14):
        assert friendly_closer_to_ball(_game(0.3, 0.2)) is True

    # Friendly regains advantage: flicker resolved, state remains True.
    assert friendly_closer_to_ball(_game(0.1, 0.4)) is True


def test_possession_transfers_after_sustained_advantage():
    assert friendly_closer_to_ball(_game(0.1, 0.4)) is True

    # 14 ticks: still held
    for _ in range(14):
        assert friendly_closer_to_ball(_game(0.3, 0.2)) is True

    # 15th tick: sustained advantage completes turnover
    assert friendly_closer_to_ball(_game(0.3, 0.2)) is False


def test_possession_breaks_immediately_on_large_clearance():
    assert friendly_closer_to_ball(_game(0.1, 0.4)) is True
    # Ball cleared far away to enemy (gap 1.0 m > 0.8 m break margin)
    assert friendly_closer_to_ball(_game(1.5, 0.5)) is False


def test_dribbler_contact_overrides_immediately():
    # Enemy currently holds distance possession
    assert friendly_closer_to_ball(_game(0.5, 0.2)) is False
    # Friendly makes physical contact with dribbler -> immediate True
    assert friendly_closer_to_ball(_game(0.5, 0.2, friendly_has_ball=True)) is True
    # Enemy makes physical contact -> immediate False
    assert friendly_closer_to_ball(_game(0.2, 0.5, enemy_has_ball=True)) is False
