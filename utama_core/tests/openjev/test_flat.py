"""Flat OpenJev engine, model-free: cells, each robot's menu, the scripted rule, the chained
decision with a fake client (the real one needs a FOR-Engine server), and a short rsim match."""

from __future__ import annotations

from utama_core.entities.data.vector import Vector2D
from utama_core.strategy.openjev import flat_state as fs
from utama_core.tests.fixtures.game_builder import build_game, make_robot


def test_cells_are_named_from_our_point_of_view():
    # A team on the left attacks +x, where its left is +y; a team on the right attacks -x, left is -y.
    left_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=False)
    right_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=True)
    assert fs.cell_of(left_team, Vector2D(4.0, 2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, -2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, 2.0)) == "attacking_right"
    assert fs.cell_of(right_team, Vector2D(4.0, 0.0)) == "defensive_centre"
