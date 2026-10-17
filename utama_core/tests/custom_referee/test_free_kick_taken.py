"""SSL rulebook §5.4 / §5.3.3: a free kick is taken, and the ball in play, once the ball has moved
0.05 m. A kick during the `DIRECT_FREE_*` preparation used to go unnoticed until the referee's own
auto-advance, so a defender the ball rolled onto was still charged with keep_out.
"""

from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


def _frame(ts: float, ball_xy: tuple[float, float]) -> GameFrame:
    zero = Vector3D(0, 0, 0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zero, a=zero),
        referee=None,
    )


def _machine_in_free_kick(ball_xy: tuple[float, float], t: float = 0.0) -> GameFrame:
    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.DIRECT_FREE_YELLOW, t)
    sm.step(t, None, _frame(t, ball_xy))
    return sm


def test_ball_moved_4cm_during_free_kick_is_still_the_restart():
    sm = _machine_in_free_kick((1.0, 0.5))
    result = sm.step(0.1, None, _frame(0.1, (1.04, 0.5)))
    assert result.referee_command == RefereeCommand.DIRECT_FREE_YELLOW


def test_ball_moved_6cm_during_free_kick_is_live_play():
    sm = _machine_in_free_kick((1.0, 0.5))
    result = sm.step(0.1, None, _frame(0.1, (1.06, 0.5)))
    assert result.referee_command == RefereeCommand.NORMAL_START


def test_free_kick_origin_is_where_the_ball_was_when_this_free_kick_began():
    """A later free kick measures its 0.05 m from the ball's spot at its own start, not the
    spot of an earlier one."""
    sm = _machine_in_free_kick((1.0, 0.5))
    sm.set_command(RefereeCommand.STOP, 0.2)
    sm.step(0.2, None, _frame(0.2, (1.5, 0.5)))
    sm.set_command(RefereeCommand.DIRECT_FREE_BLUE, 0.3)
    sm.step(0.3, None, _frame(0.3, (1.5, 0.5)))
    result = sm.step(0.4, None, _frame(0.4, (1.54, 0.5)))
    assert result.referee_command == RefereeCommand.DIRECT_FREE_BLUE
