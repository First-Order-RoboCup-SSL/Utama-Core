"""Regression test for `BlockShapeTactic`'s `go_to_point()` call sites passing
a raw `(x, y)` tuple instead of a `Vector2D`.

Commit c4ad99c ("Fix two bugs blocking trajsample from playing a real
match"): `go_to_point()`'s signature declares `target_coords: Vector2D`, and
`move()` forwards that value straight through to
`motion_controller.calculate(..., target_pos=target_coords, ...)`
unconverted (`utama_core/skills/src/utils/move_utils.py`). FPP/DWA happened
to tolerate a raw tuple silently; trajsampling's stricter `target_pos.x`/`.y`
attribute access crashed outright — the actual reason a full trajsample
tournament scored 0-0 everywhere. Both of `BlockShapeTactic.tick()`'s
`go_to_point()` call sites (the stepped-out presser, and each screen-line
robot) used to pass a raw `(lead_x, lead_y)` / `(screen_x, target_y)` tuple;
the fix wraps both in `Vector2D(...)`.

This test does not need a real trajsample planner to pin the boundary: any
motion controller whose `calculate()` accesses `target_pos.x`/`.y` (as
trajsampling's real one does) is sufficient to distinguish "received a
Vector2D" from "received a bare tuple" — a tuple has no `.x`/`.y` attributes
and raises `AttributeError` on first access, exactly reproducing the crash
this commit fixed, without needing the planner itself.
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
from utama_core.tactics.block_shape import BlockShapeTactic


class _AttrAccessMotionController(MotionController):
    """Mimics trajsampling's stricter `target_pos.x`/`.y` attribute access
    (as opposed to FPP/DWA, which tolerated an indexable/tuple `target_pos`
    silently) -- raises `AttributeError` if `target_pos` is a bare tuple
    rather than a `Vector2D`, exactly reproducing the crash this commit
    fixed without needing the real trajsample planner."""

    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        _ = target_pos.x, target_pos.y  # raises AttributeError on a raw tuple
        return Vector2D(0.0, 0.0), 0.0


def _make_game(presser_pos: Vector2D, screen_pos: Vector2D, ball_pos: Vector2D, ball_v: Vector2D) -> Game:
    friendly = {
        1: Robot(
            id=1, is_friendly=True, has_ball=False, p=presser_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
        2: Robot(
            id=2, is_friendly=True, has_ball=False, p=screen_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    zv = Vector3D(0, 0, 0)
    ball = Ball(Vector3D(ball_pos.x, ball_pos.y, 0.0), Vector3D(ball_v.x, ball_v.y, 0.0), zv)
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots={}, ball=ball
    )
    field = Field(
        my_team_is_right=True,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_presser_go_to_point_receives_a_vector2d_not_a_raw_tuple():
    """The stepped-out first-defender `go_to_point()` call site (the ball is
    not loose here, so the tactic takes the lead-offset branch, not the
    go_to_ball branch) must pass a `Vector2D`, not a raw `(lead_x, lead_y)`
    tuple -- a strict-attribute-access motion controller must not raise."""
    # Ball not loose: give it clear nonzero velocity so ball_is_loose()
    # takes the "carrier" branch (lead-offset go_to_point), which is the
    # exact call site c4ad99c fixed.
    presser_pos = Vector2D(0.0, 0.0)
    screen_pos = Vector2D(1.0, 1.0)
    ball_pos = Vector2D(2.0, 0.0)
    game = _make_game(presser_pos, screen_pos, ball_pos, ball_v=Vector2D(2.0, 0.0))  # fast -> ball_is_loose() False

    ctx = TickContext(motion_controller=_AttrAccessMotionController(), match_log=None)
    tactic = BlockShapeTactic()
    mem = tactic.initial_mem()

    # Must not raise AttributeError -- pre-fix, the presser's go_to_point()
    # call passed a raw tuple and this line crashed.
    commands, _mem = tactic.tick(game, ctx, (1, 2), mem)
    assert 1 in commands


def test_screen_go_to_point_receives_a_vector2d_not_a_raw_tuple():
    """The screen-line `go_to_point()` call site (every robot other than the
    presser) must also pass a `Vector2D`, not a raw `(screen_x, target_y)`
    tuple."""
    presser_pos = Vector2D(0.0, 0.0)
    screen_pos = Vector2D(1.0, 1.0)
    ball_pos = Vector2D(2.0, 0.0)
    game = _make_game(presser_pos, screen_pos, ball_pos, ball_v=Vector2D(2.0, 0.0))

    ctx = TickContext(motion_controller=_AttrAccessMotionController(), match_log=None)
    tactic = BlockShapeTactic()
    mem = tactic.initial_mem()

    commands, _mem = tactic.tick(game, ctx, (1, 2), mem)
    # Whichever robot isn't the presser is the screen-line robot exercising
    # the second fixed call site.
    screen_id = 1 if mem.presser_id == 2 else 2
    assert screen_id in commands
