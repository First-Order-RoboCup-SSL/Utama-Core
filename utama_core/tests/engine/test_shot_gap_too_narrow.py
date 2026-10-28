"""A shot gap too narrow for the shot gate counts as no shot, in every tactic that shoots.

`find_best_shot` returns the widest gap however narrow; `heading_on_target` (the shot gate) only
kicks with room for the aim error. Turned to a gap narrower than that, the robot held the ball
and turned on the spot forever (review of #152). Now it moves for a better shot instead.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.tactics import _pass_and_score
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic

# Our goal is at +x (my_team_is_right), the enemy goal at -x. The shooter faces it, ball on its
# dribbler, and an enemy stands 1 m ahead on the line to the goal centre.
_SHOOTER = Vector2D(-2.5, 0.0)


def _robot(rid: int, x: float, y: float, friendly: bool, has_ball: bool = False) -> Robot:
    zero = Vector2D(0.0, 0.0)
    return Robot(id=rid, is_friendly=friendly, has_ball=has_ball, p=Vector2D(x, y), v=zero, a=zero, orientation=math.pi)


def _game(blocker_y: float) -> Game:
    zv = Vector3D(0.0, 0.0, 0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, _SHOOTER.x, _SHOOTER.y, True, has_ball=True)},
        enemy_robots={0: _robot(0, _SHOOTER.x - 1.0, blocker_y, False)},
        ball=Ball(p=Vector3D(_SHOOTER.x - 0.09, _SHOOTER.y, 0.0), v=zv, a=zv),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(10), current=frame, field=field)


def _ctx() -> TickContext:
    mc = MagicMock()
    mc.calculate.return_value = (Vector2D(0.0, 0.0), 0.0)
    return TickContext(motion_controller=mc, match_log=None)


def _lead(game: Game) -> None:
    tactic = LeadAndSupportTactic()
    tactic.tick(game, _ctx(), (1,), tactic.initial_mem())


def _give_and_go(game: Game) -> None:
    tactic = GiveAndGoTactic()
    tactic.tick(game, _ctx(), (1,), tactic.initial_mem())


def _score_goal(game: Game) -> None:
    _pass_and_score._score_goal(game, _ctx(), 1)


_TACTICS = {
    "lead_and_support": ("utama_core.tactics.lead_and_support", _lead),
    "give_and_go": ("utama_core.tactics.give_and_go", _give_and_go),
    "_score_goal": ("utama_core.tactics._pass_and_score", _score_goal),
}


@pytest.mark.parametrize("name", list(_TACTICS))
@pytest.mark.parametrize("blocker_y, turns", [(0.05, False), (0.6, True)])
def test_a_shooter_moves_for_a_better_shot_when_the_best_gap_is_too_narrow(name, blocker_y, turns):
    """`find_best_shot` says aim at the goal centre. With the blocker 5 cm off that line the
    gate can never open: the shooter repositions (`move`), it does not turn to it forever. With
    the blocker 0.6 m off the line the gap is wide: it turns to shoot as before."""
    module, run = _TACTICS[name]
    with (
        patch(f"{module}.find_best_shot", return_value=(0.0, 1.0)),
        patch(f"{module}.turn_on_spot") as turn,
        patch(f"{module}.move") as move,
        patch(f"{module}.kick") as kick,
    ):
        run(_game(blocker_y))
    if turns:
        assert turn.called or kick.called
        assert not move.called
    else:
        assert move.called
        assert not turn.called and not kick.called
