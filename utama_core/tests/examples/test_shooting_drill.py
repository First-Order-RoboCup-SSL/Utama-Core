"""`examples/real_attacker_defender.py`'s demo keeper: it keeps goal, and a ball it holds goes back
to the attacker, so the demo loops."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from utama_core.config.enums import Mode
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import real_attacker_defender as ex
finally:
    sys.path.remove(str(EXAMPLES))


@pytest.fixture(autouse=True)
def _fresh_possession_state():
    reset_possession_state()
    yield
    reset_possession_state()


class _StillController(MotionController):
    def __init__(self):
        super().__init__(mode="rsim")
        self.mode = Mode.RSIM  # the keeper builds its own PIDController from it

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0


def _tick(keeper_orientation: float, has_ball: bool = True):
    """Keeper at its goal (+x, our side is right), attacker 1.5 m upfield at (0.0, 0.3)."""
    keeper = make_robot(0, is_friendly=True, x=1.3, y=0.0, orientation=keeper_orientation, has_ball=has_ball)
    attacker = make_robot(0, is_friendly=False, x=-0.2, y=0.3)
    ball = make_ball(x=1.3 + 0.09 * math.cos(keeper_orientation), y=0.09 * math.sin(keeper_orientation))
    game = build_game({0: keeper}, {0: attacker}, ball, my_team_is_right=True, field_dims=ex.FIELD)
    commands, _ = ex.FeedingKeeperTactic().tick(game, TickContext(motion_controller=_StillController()), (0,), None)
    return commands[0]


_TO_ATTACKER = math.atan2(0.3, -1.5)


@pytest.mark.parametrize("off, kicks", [(ex.FEED_TOLERANCE_RAD - 0.01, True), (ex.FEED_TOLERANCE_RAD + 0.01, False)])
def test_the_keeper_kicks_a_held_ball_once_facing_the_attacker(off, kicks):
    """Upfield (pi) is 0.2 rad from the attacker here: the old clearance would kick out of play."""
    command = _tick(_TO_ATTACKER + off)
    assert bool(command.kick) is kicks
    if not kicks:
        assert command.dribble  # turns with the ball on its dribbler


def test_without_the_ball_it_keeps_goal():
    command = _tick(math.pi, has_ball=False)
    assert not command.kick
