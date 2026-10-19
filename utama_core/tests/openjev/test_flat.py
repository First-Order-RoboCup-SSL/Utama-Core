"""Flat OpenJev engine, model-free: cells, each robot's menu, the scripted rule, the chained
decision with a fake client (the real one needs a FOR-Engine server), and a short rsim match."""

from __future__ import annotations

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.openjev import flat_state as fs
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

OUTFIELD = (1, 2, 3, 4, 5)


@pytest.mark.parametrize("is_right", [True, False])
@pytest.mark.parametrize("cell", fs.CELLS)
def test_every_cell_centre_maps_back_to_its_cell(is_right, cell):
    game = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=is_right)
    x_lo, x_hi, y_lo, y_hi = fs.cell_bounds(game, cell)
    assert fs.cell_of(game, Vector2D((x_lo + x_hi) / 2, (y_lo + y_hi) / 2)) == cell


def test_cells_are_named_from_our_point_of_view():
    # A team on the left attacks +x, where its left is +y; a team on the right attacks -x, left is -y.
    left_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=False)
    right_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=True)
    assert fs.cell_of(left_team, Vector2D(4.0, 2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, -2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, 2.0)) == "attacking_right"
    assert fs.cell_of(right_team, Vector2D(4.0, 0.0)) == "defensive_centre"


def test_adjacent_cells_include_the_cell_itself():
    assert fs.adjacent_cells("defensive_left") == ["defensive_left", "defensive_centre", "middle_left", "middle_centre"]
    assert len(fs.adjacent_cells("middle_centre")) == 9
    assert len(fs.adjacent_cells("attacking_centre")) == 6


def _game(carrier_xy=None, mates=None, enemies=None, ball_xy=(0.0, 0.0)):
    """Our team on the left (attacking +x), referee live so the enemy keeper id (0) is known."""
    friendly = {0: make_robot(0, is_friendly=True, x=-4.4, y=0.0)}
    for rid, (x, y) in (mates or {}).items():
        friendly[rid] = make_robot(rid, is_friendly=True, x=x, y=y)
    has_ball_robot = None
    if carrier_xy is not None:
        rid, x, y = carrier_xy
        friendly[rid] = make_robot(rid, is_friendly=True, x=x, y=y)
        ball_xy = (x + 0.09, y)
        has_ball_robot = ("friendly", rid)
    enemy = {eid: make_robot(eid, is_friendly=False, x=x, y=y) for eid, (x, y) in (enemies or {0: (4.4, 0.0)}).items()}
    return build_game(
        friendly,
        enemy,
        make_ball(*ball_xy),
        my_team_is_right=False,
        referee_command=RefereeCommand.NORMAL_START,
        has_ball_robot=has_ball_robot,
    )


def _view(game, rid, carry_left_m=None, marked_by=None, blockers=0):
    return fs.robot_view(game, rid, OUTFIELD, carry_left_m=carry_left_m, marked_by=marked_by or {}, blockers=blockers)


def test_carrier_menu_has_ball_actions_and_no_dribble_once_its_carry_is_used():
    game = _game(carrier_xy=(2, 0.0, 0.0), mates={3: (1.0, 2.0), 4: (-1.0, 0.0)})
    view = _view(game, 2, carry_left_m=0.5)
    assert view.has_ball
    assert {"shoot", "pass:3", "pass:4", "clear"} <= set(view.options)
    assert "pass:0" not in view.options  # the goalkeeper plays by itself
    assert any(k.startswith("dribble:") for k in view.options)
    assert not any(k.startswith("dribble:") for k in _view(game, 2, carry_left_m=0.2).options)


def test_off_ball_menu_marks_every_opponent_but_their_goalkeeper():
    game = _game(mates={1: (0.0, 0.0)}, enemies={0: (4.4, 0.0), 3: (1.0, 1.0), 5: (-2.0, -1.0)}, ball_xy=(1.0, 0.0))
    view = _view(game, 1, marked_by={3: [4]}, blockers=1)
    assert not view.has_ball
    assert {"get_ball", "block_shot", "hold", "mark:3", "mark:5"} <= set(view.options)
    assert "mark:0" not in view.options
    assert len([k for k in view.options if k.startswith("move:")]) == 9
    assert view.options["mark:3"].endswith("marked by 4")
    assert view.options["mark:5"].endswith("unmarked")
