"""End-to-end tests for the foul-counter / yellow-card mechanism
(SSL rulebook §8.4: "every third increase to the foul counter causes a
yellow card to be awarded").

`RuleViolation.offending_teams`/`counts_toward_foul_counter` and
`TeamInfo.increment_foul_counter()` are each unit-tested in isolation
elsewhere, but nothing drives real fouls through `GameStateMachine.step()`
in sequence and asserts a yellow card actually lands on
`TeamInfo.yellow_cards` — see `docs/testing_gaps.md` gap #2. This file
closes that gap by driving `GameStateMachine.step()` itself (not
`TeamInfo.increment_foul_counter()` or `GameStateMachine._handle_foul()`
directly), the same real call path `CustomReferee.step()` uses in
production.
"""

from __future__ import annotations

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.referee.referee_command import RefereeCommand

# GameStateMachine._TRANSITION_COOLDOWN is 0.3s — a stopping foul's
# transition is suppressed if it arrives sooner than that after the last
# one (see _apply_violation/_can_transition), so successive stopping
# violations in these tests are spaced 1.0s apart to stay well clear of it.
_STEP = 1.0


def _state_machine() -> GameStateMachine:
    sm = GameStateMachine(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        n_robots_yellow=3,
        n_robots_blue=3,
    )
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.NORMAL_START, 0.0)
    return sm


def _foul_violation(offending_teams: tuple[bool, ...], counts: bool = True) -> RuleViolation:
    """A generic stopping foul shaped like most §8.4.1 rules (e.g.
    BallSpeedRule) — only the foul-counter-relevant fields matter here."""
    return RuleViolation(
        rule_name="test_foul",
        suggested_command=RefereeCommand.STOP,
        next_command=RefereeCommand.DIRECT_FREE_BLUE,
        status_message="Test foul",
        offending_teams=offending_teams,
        counts_toward_foul_counter=counts,
    )


class TestFoulCounterYellowCard:
    def test_third_foul_awards_yellow_card_to_yellow_team(self):
        sm = _state_machine()
        t = 1.0

        sm.step(current_time=t, violation=_foul_violation((True,)))
        assert sm.yellow_team.foul_counter == 1
        assert sm.yellow_team.yellow_cards == 0

        t += _STEP
        sm.step(current_time=t, violation=_foul_violation((True,)))
        assert sm.yellow_team.foul_counter == 2
        assert sm.yellow_team.yellow_cards == 0

        t += _STEP
        sm.step(current_time=t, violation=_foul_violation((True,)))
        assert sm.yellow_team.foul_counter == 3
        assert sm.yellow_team.yellow_cards == 1

        # Blue must be completely untouched throughout.
        assert sm.blue_team.foul_counter == 0
        assert sm.blue_team.yellow_cards == 0

    def test_third_foul_awards_yellow_card_to_blue_team(self):
        """Not yellow-hardcoded: offending_teams=(False,) charges blue."""
        sm = _state_machine()
        t = 1.0

        for _ in range(2):
            sm.step(current_time=t, violation=_foul_violation((False,)))
            t += _STEP
        assert sm.blue_team.foul_counter == 2
        assert sm.blue_team.yellow_cards == 0

        sm.step(current_time=t, violation=_foul_violation((False,)))
        assert sm.blue_team.foul_counter == 3
        assert sm.blue_team.yellow_cards == 1

        assert sm.yellow_team.foul_counter == 0
        assert sm.yellow_team.yellow_cards == 0

    def test_sixth_foul_awards_a_second_yellow_card(self):
        """ "Every third increase" is not a one-shot special case for
        exactly 3 — a 4th/5th/6th foul must award a 2nd card on the 6th."""
        sm = _state_machine()
        t = 1.0
        for _ in range(6):
            sm.step(current_time=t, violation=_foul_violation((True,)))
            t += _STEP
        assert sm.yellow_team.foul_counter == 6
        assert sm.yellow_team.yellow_cards == 2

    def test_carve_out_violation_does_not_increment_foul_counter(self):
        """Mirrors DefenseAreaRule's Multiple Defenders carve-out
        (counts_toward_foul_counter=False even though a team is at fault,
        per the rulebook's explicit exception)."""
        sm = _state_machine()
        sm.step(current_time=1.0, violation=_foul_violation((True,), counts=False))
        assert sm.yellow_team.foul_counter == 0
        assert sm.yellow_team.yellow_cards == 0
        assert sm.blue_team.foul_counter == 0

    def test_no_fault_violation_charges_neither_team(self):
        """Mirrors PushingRule's symmetric-force branch: offending_teams=()
        when "no team is at fault" per the rulebook — must not silently
        charge anyone."""
        sm = _state_machine()
        sm.step(current_time=1.0, violation=_foul_violation(()))
        assert sm.yellow_team.foul_counter == 0
        assert sm.yellow_team.yellow_cards == 0
        assert sm.blue_team.foul_counter == 0
        assert sm.blue_team.yellow_cards == 0

    def test_non_stopping_foul_still_increments_foul_counter(self):
        """A non-stopping foul (e.g. Crashing, is_stopping=False) must still
        apply its foul-counter side effect even though it doesn't touch
        command/next_command — "the game continues normally" describes the
        command transition, not the foul-counter bookkeeping."""
        sm = _state_machine()
        violation = RuleViolation(
            rule_name="test_non_stopping_foul",
            suggested_command=RefereeCommand.NORMAL_START,
            next_command=None,
            status_message="Non-stopping foul",
            offending_teams=(True,),
            counts_toward_foul_counter=True,
            is_stopping=False,
        )
        prev_command = sm.command
        data = sm.step(current_time=1.0, violation=violation)
        assert sm.yellow_team.foul_counter == 1
        # Command must be left untouched — "the game continues normally".
        assert sm.command == prev_command
        assert data.referee_command == prev_command

    def test_non_stopping_fouls_do_not_consume_the_transition_cooldown(self):
        """Non-stopping fouls never update _last_transition_time (see
        _apply_violation), so back-to-back non-stopping violations are
        never suppressed by the 0.3s cooldown the way stopping fouls are."""
        sm = _state_machine()

        def _non_stopping(offending_teams):
            return RuleViolation(
                rule_name="test_non_stopping_foul",
                suggested_command=RefereeCommand.NORMAL_START,
                next_command=None,
                status_message="Non-stopping foul",
                offending_teams=offending_teams,
                is_stopping=False,
            )

        # Same instant, no cooldown spacing at all.
        sm.step(current_time=1.0, violation=_non_stopping((True,)))
        sm.step(current_time=1.0, violation=_non_stopping((True,)))
        sm.step(current_time=1.0, violation=_non_stopping((True,)))
        assert sm.yellow_team.foul_counter == 3
        assert sm.yellow_team.yellow_cards == 1
