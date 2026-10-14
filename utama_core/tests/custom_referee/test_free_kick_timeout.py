"""SSL rulebook §5.4: the ball is in play once "5 seconds (Division A) or 10 seconds
(Division B) passed following a free kick". Our DIRECT_FREE_* waited for the kicker
with no limit, so a free kick nobody took held the game forever.
"""

from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


def _frame(ts: float) -> GameFrame:
    zero = Vector3D(0, 0, 0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={},  # no kicker, so the free kick is never "ready"
        enemy_robots={},
        ball=Ball(p=Vector3D(1.0, 0.5, 0), v=zero, a=zero),
        referee=None,
    )


def test_untaken_free_kick_is_in_play_after_ten_seconds():
    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)
    assert sm.step(9.9, None, _frame(9.9)).referee_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert sm.step(10.0, None, _frame(10.0)).referee_command == RefereeCommand.FORCE_START


def test_defender_too_close_restarts_the_free_kick_clock():
    """§8.4.3: on a Defender Too Close foul "the timer of the opponent team for
    bringing the ball into play is reset"."""
    from utama_core.custom_referee.rules.base_rule import RuleViolation

    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)
    keep_out = RuleViolation(
        rule_name="keep_out",
        suggested_command=RefereeCommand.DIRECT_FREE_YELLOW,
        next_command=None,
        status_message="Defender too close to ball",
        offending_teams=(False,),
        is_stopping=False,
    )
    sm.step(6.0, keep_out, _frame(6.0))
    assert sm.step(15.9, None, _frame(15.9)).referee_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert sm.step(16.0, None, _frame(16.0)).referee_command == RefereeCommand.FORCE_START


def test_only_the_first_defender_too_close_foul_restarts_the_free_kick_clock():
    """A defender parked inside 0.5 m re-raises keep_out every 2 s; when each foul
    restarted the clock, the free kick was never forced and held the rest of a match."""
    from utama_core.custom_referee.rules.base_rule import RuleViolation

    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)
    keep_out = RuleViolation(
        rule_name="keep_out",
        suggested_command=RefereeCommand.DIRECT_FREE_YELLOW,
        next_command=None,
        status_message="Defender too close to ball",
        offending_teams=(False,),
        is_stopping=False,
    )
    sm.step(2.0, keep_out, _frame(2.0))
    for t in (4.0, 6.0, 8.0, 10.0):
        assert sm.step(t, keep_out, _frame(t)).referee_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert sm.step(11.9, None, _frame(11.9)).referee_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert sm.step(12.0, None, _frame(12.0)).referee_command == RefereeCommand.FORCE_START
