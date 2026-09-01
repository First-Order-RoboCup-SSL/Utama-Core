"""Regression test for `GiveAndGoTactic`'s first-touch `hop_ticks` timeout.

Found live in a tournament replay (`counter_flow_vs_counter_press.pkl`,
2026-09-01): a carrier with no reachable/unblocked teammate held the ball
for 11+ seconds, far past the intended 4s (`_MAX_HOP_TICKS`) abandon budget.
Root cause: the first-touch-of-possession branch in `tick()` unconditionally
reset `mem.hop_ticks = 0` right before incrementing it whenever
`_nearest_safe_receiver` was called, even when it returned `None` — pinning
`hop_ticks` at exactly 1 forever instead of climbing toward `_MAX_HOP_TICKS`.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.give_and_go import _MAX_HOP_TICKS, GiveAndGoTactic


class _NullMotionController(MotionController):
    """Never actually reached by this test's carrier (no receiver is ever
    locked), but `_relocate_others`' support-repositioning calls into it."""

    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0


def _make_game(carrier_pos: Vector2D, teammate_pos: Vector2D, enemy_pos: Vector2D) -> Game:
    friendly = {
        1: Robot(
            id=1, is_friendly=True, has_ball=True, p=carrier_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
        2: Robot(
            id=2, is_friendly=True, has_ball=False, p=teammate_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    enemy = {
        3: Robot(
            id=3, is_friendly=False, has_ball=False, p=enemy_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    ball = Ball(Vector3D(carrier_pos.x, carrier_pos.y, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots=enemy, ball=ball
    )
    field = Field(
        my_team_is_right=True,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_hop_ticks_climbs_when_no_receiver_is_ever_reachable():
    """With the teammate's only lane blocked by an enemy standing on it,
    `_nearest_safe_receiver` returns None every tick. `hop_ticks` must climb
    toward `_MAX_HOP_TICKS` instead of being reset to 0 every tick it fails.
    """
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(1.0, 0.0)
    enemy_pos = Vector2D(0.5, 0.0)  # sits directly on the carrier->teammate segment

    game = _make_game(carrier_pos, teammate_pos, enemy_pos)
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    for _ in range(10):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)
        assert mem.receiver_id is None  # lane stays blocked the whole test

    # Before the fix, hop_ticks was reset to 0 then incremented to 1 on every
    # tick, so after 10 ticks it would still read 1.
    assert mem.hop_ticks == 10


def test_carrier_eventually_abandons_the_hold_after_max_hop_ticks():
    """Once hop_ticks reaches _MAX_HOP_TICKS, the carrier must fall through to
    the shoot/reposition branch (receiver_id stays None, hop_ticks resets to
    0) rather than holding forever."""
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(1.0, 0.0)
    enemy_pos = Vector2D(0.5, 0.0)

    game = _make_game(carrier_pos, teammate_pos, enemy_pos)
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    for _ in range(_MAX_HOP_TICKS):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)

    assert mem.hop_ticks < _MAX_HOP_TICKS
    assert mem.receiver_id is None
