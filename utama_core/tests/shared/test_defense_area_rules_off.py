"""`Field.defense_area_rules=False`, from a referee profile without the defense-area rule (real
play with two or three robots, 2026-10-28): the opponent's box stops nothing, and a target in our
own box is left where it is."""

from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee import actions
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.motion_planning.src.trajsampling.planner import (
    TrajectorySamplingPlanner,
)
from utama_core.shared.pass_and_score_geometry import (
    ball_in_enemy_defense_area,
    ball_in_own_defense_area,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
)

# We are on the right: our box spans x >= 3.5, the opponent's x <= -3.5 (standard field).
_ENEMY_BOX = Vector2D(-4.0, 0.0)
_OWN_BOX = Vector2D(4.0, 0.0)


def _game(rules: bool, ball: Vector2D = _ENEMY_BOX) -> Game:
    dims = STANDARD_FIELD_DIMS
    zero = Vector2D(0, 0)
    robot = Robot(id=0, is_friendly=True, has_ball=False, p=Vector2D(0.0, 0.0), v=zero, a=zero, orientation=0.0)
    frame = GameFrame(
        ts=1.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball.x, ball.y, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        referee=SimpleNamespace(referee_command=RefereeCommand.FORCE_START, designated_position=None),
    )
    return Game(past=GameHistory(10), current=frame, field=Field(True, dims, dims.full_field_bounds, rules))


@pytest.mark.parametrize("rules", [True, False])
def test_the_opponents_box_is_only_a_box_while_its_rules_apply(rules):
    game = _game(rules)
    assert ball_in_enemy_defense_area(game) is rules
    assert (clamp_outside_enemy_defense_area(game, _ENEMY_BOX) == _ENEMY_BOX) is not rules
    assert (actions._project_outside_opp_defense_area(game, _ENEMY_BOX, 0.2) == _ENEMY_BOX) is not rules
    assert actions._in_planner_opp_defense_margin(game, _ENEMY_BOX) is rules


@pytest.mark.parametrize("rules", [True, False])
def test_a_target_in_our_own_box_stays_put_without_the_rules(rules):
    game = _game(rules)
    assert (clamp_outside_own_defense_area(game, _OWN_BOX) == _OWN_BOX) is not rules
    assert actions._is_in_own_defense_area(game, _OWN_BOX.x, _OWN_BOX.y) is rules


def test_a_ball_resting_in_our_own_box_is_still_the_keepers():
    assert ball_in_own_defense_area(_game(False, ball=_OWN_BOX))


@pytest.mark.parametrize(
    "planner", [FastPathPlanner(env=None), TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)], ids=["fpp", "trajsampling"]
)
@pytest.mark.parametrize("rules", [True, False])
def test_the_planners_route_through_the_opponents_box_in_open_play_without_the_rules(planner, rules):
    assert planner._enemy_defense_area_retrieval_exempt(_game(rules), 0) is not rules


@pytest.mark.parametrize("profile, rules", [("simulation", True), ("human", False)])
def test_the_runner_takes_the_box_rules_from_the_referee_profile(profile, rules):
    """`human` (and the real examples, built on it) has no defense-area rule, so robots ignore the boxes."""
    from utama_core.custom_referee import CustomReferee
    from utama_core.run.strategy_runner import StrategyRunner

    runner = SimpleNamespace(referee=CustomReferee.from_profile_name(profile))
    assert StrategyRunner._defense_area_rules(runner) is rules


def test_without_our_referee_the_box_rules_apply():
    from utama_core.run.strategy_runner import StrategyRunner

    assert StrategyRunner._defense_area_rules(SimpleNamespace(referee=None)) is True
