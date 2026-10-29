"""The striker's starts fit the selected pitch, and deployment uses the training frame and clock."""

import math

import numpy as np
import pytest

from tools import striker_duel as duel
from utama_core.config.field_params import (
    GREAT_EXHIBITION_FIELD_DIMS,
    STANDARD_FIELD_DIMS,
    FieldBounds,
)
from utama_core.engine.context import TickContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game.field import Field
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot


def _game(ts=0.0, right=False):
    sign = -1 if right else 1
    return build_game(
        {3: make_robot(3, is_friendly=True, x=sign * 0.2, y=sign * 0.1, orientation=math.pi if right else 0.0)},
        {1: make_robot(1, is_friendly=False, x=sign * 1.35, y=sign * 0.1)},
        make_ball(sign * 0.31, sign * 0.1),
        has_ball_robot=("friendly", 3),
        field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        my_team_is_right=right,
        ts=ts,
    )


def test_a_live_striker_uses_elapsed_play_time_and_resets_it_with_its_memory(monkeypatch):
    observations = []

    def choose(obs, mask):
        observations.append(obs)
        return duel.N_SHOTS, 0.0

    tactic = duel.StrikerTactic(choose, robot_id=3)
    monkeypatch.setattr(tactic, "_execute", lambda *args: RobotCommand(0, 0, 0, False, False, False))
    mem = tactic.initial_mem()
    for ts in (10000.0, 10004.0, 10012.0):
        _, mem = tactic.tick(_game(ts), TickContext(motion_controller=None), (3,), mem)
    assert [obs[18] for obs in observations] == pytest.approx([0.0, 0.5, 1.0])
    tactic.tick(_game(10020.0), TickContext(motion_controller=None), (3,), tactic.initial_mem())
    assert observations[-1][18] == 0.0


def test_policy_inputs_and_shot_indices_rotate_with_the_team():
    left, right = _game(), _game(right=True)
    np.testing.assert_allclose(
        duel.observe(left, 3, duel.StrikerMem(), 0.0),
        duel.observe(right, 3, duel.StrikerMem(), 0.0),
        atol=1e-6,
    )
    for i in range(duel.N_SHOTS):
        a, b = duel.shot_point(left, i), duel.shot_point(right, i)
        assert (a.x, a.y) == pytest.approx((-b.x, -b.y))


def test_the_automatic_duel_rejects_real_mode_before_opening_controllers():
    with pytest.raises(ValueError, match="operator-controlled"):
        duel.Duel(mode="real")


def test_a_striker_holds_when_the_keeper_is_not_seen():
    game = _game()
    game.enemy_robots.clear()
    tactic = duel.StrikerTactic(robot_id=3)
    commands, _ = tactic.tick(game, TickContext(motion_controller=None), (3,), tactic.initial_mem())
    assert commands == {}


def test_a_camera_half_does_not_change_the_policys_coordinate_scale():
    full, half = _game(), _game()
    dims = GREAT_EXHIBITION_FIELD_DIMS
    half._field = Field(False, dims, FieldBounds((0.0, 1.125), (1.5, -1.125)))
    np.testing.assert_allclose(
        duel.observe(full, 3, duel.StrikerMem(), 0.0),
        duel.observe(half, 3, duel.StrikerMem(), 0.0),
    )


def test_dribbling_in_the_camera_half_uses_its_actual_offset_bounds():
    game = _game()
    game._field = Field(False, STANDARD_FIELD_DIMS, FieldBounds((0.0, 3.0), (4.5, -3.0)))
    game.friendly_robots[3] = make_robot(3, is_friendly=True, x=2.2)
    target = duel.dribble_target(game, 3, 0)
    # A legal forward carry used to be clamped back to x=1.95: half_length describes
    # the camera view's extent, but the view is centred at x=2.25 rather than x=0.
    assert target.x == pytest.approx(2.55)
    assert target.y == 0.0


def test_a_gerf_carry_keeps_the_margin_at_the_camera_boundary():
    game = _game()
    game._field = Field(False, GREAT_EXHIBITION_FIELD_DIMS, FieldBounds((0.0, 1.125), (1.5, -1.125)))
    game.friendly_robots[3] = make_robot(3, is_friendly=True, x=0.4)
    target = duel.dribble_target(game, 3, 180)
    assert target.x == pytest.approx(0.3)
