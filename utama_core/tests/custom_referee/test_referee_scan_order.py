"""Tests for `CustomReferee.step()`'s stopping/non-stopping scan-order logic
(`docs/testing_gaps.md` gap #3).

The scan (see `custom_referee.py::step()`) walks `self._rules` in priority
order (`_build_active_rules()`): the first *stopping* violation found wins
and stops the scan (`break`) immediately; a *non-stopping* violation found
earlier does not get discarded by a later stopping one, but also can't
pre-empt it — a non-stopping violation is only ever applied when the whole
scan finds no stopping violation at all. This logic was added specifically
so `CrashingRule` (`is_stopping=False`) coexists with every pre-existing
rule (`is_stopping=True` by default) without regressing any of them, and had
no dedicated test of its own before this file.

Under the real profile's rule ordering and command-gating, a non-stopping
violation found *earlier* in list order than a same-tick stopping one can't
actually arise from two real rules: every rule that shares `CrashingRule`'s
`NORMAL_START`/`FORCE_START` gate (`GoalRule`, `OutOfBoundsRule`,
`BallSpeedRule`, `DoubleTouchRule`, `DefenseAreaRule`, `KeepOutRule`,
`PushingRule`) sits earlier than it in `_build_active_rules()`'s list, and
every stopping rule positioned after it (`DefenseAreaStoppageRule`,
`BallPlacementInterferenceRule`) only fires during stoppage commands
Crashing itself never checks. So `test_earlier_non_stopping_does_not_block_a_later_stopping_violation`
below uses two minimal stub `BaseRule`s to exercise the scan mechanism
itself in that ordering — the one combination the real rule set can't
currently produce, but which the loop must still get right. The other two
tests use real, unmodified rule instances end to end.
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

# Robots at PushingRule/CrashingRule's contact distance (2*ROBOT_RADIUS+0.03 =
# 0.18m), same convention as test_pushing_crashing.py.
_CONTACT_X = 0.18


def _ball(x: float = 5.0, y: float = 5.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


def _robot(robot_id: int, x: float, y: float, is_friendly: bool, vx: float = 0.0, vy: float = 0.0) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(
    friendly: Robot,
    enemy: Robot,
    ball: Optional[Ball] = None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
    ts: float = 10.0,
) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots={friendly.id: friendly},
        enemy_robots={enemy.id: enemy},
        ball=ball if ball is not None else _ball(),
        referee=None,
    )


class _StubRule(BaseRule):
    """Returns a fixed `RuleViolation` once `armed`, then stays armed
    (does not self-disarm) — a spy-friendly stand-in for exercising the
    scan's ordering logic without any of a real rule's own trigger geometry
    or persistence bookkeeping getting in the way."""

    def __init__(self, violation: RuleViolation) -> None:
        self._violation = violation
        self.armed = True
        self.call_count = 0

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        self.call_count += 1
        return self._violation if self.armed else None


class TestScanOrderWithRealRules:
    def test_crashing_alone_is_recorded_but_does_not_change_command(self):
        """A non-stopping violation (Crashing) firing with nothing stopping
        alongside it must still be recorded as `last_violation` (its
        foul-counter side effect must apply), but `referee_command` must be
        completely unchanged from before this tick — a non-stopping foul
        "continues normally" (SSL rulebook §8.4.2), it does not freeze play
        or transition the command the way every stopping foul does."""
        referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
        referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

        # First tick: robots far apart, ball safely in-field, nothing fires —
        # establishes a stable NORMAL_START baseline before the crash happens.
        far_friendly = _robot(0, -2.0, 0.0, True, vx=0.0)
        far_enemy = _robot(0, 2.0, 0.0, False, vx=0.0)
        in_field_ball = _ball(x=0.0, y=0.0)
        baseline = referee.step(_frame(far_friendly, far_enemy, ball=in_field_ball, ts=0.0), current_time=0.0)
        assert baseline.referee_command == RefereeCommand.NORMAL_START

        # Second tick: friendly crashes into enemy at fault speed (rising-edge
        # contact — CrashingRule fires immediately, no persistence needed).
        # Ball position is irrelevant to CrashingRule, but must stay in-field
        # so OutOfBoundsRule doesn't also fire and confound the assertion.
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        result = referee.step(_frame(friendly, enemy, ball=in_field_ball, ts=1.0), current_time=1.0)

        assert referee.last_violation is not None
        assert referee.last_violation.rule_name == "crashing"
        assert referee.last_violation.is_stopping is False
        # Command must be untouched — still NORMAL_START, not overwritten
        # with the non-stopping violation's (ignored) suggested_command.
        assert result.referee_command == RefereeCommand.NORMAL_START

    def test_first_stopping_rule_in_priority_order_wins_and_later_rules_are_never_consulted(self):
        """`GoalRule` sits first in `_build_active_rules()`'s priority order,
        `OutOfBoundsRule` second. A ball resting inside the goal is a goal —
        `GoalRule` must fire and stop the scan via `break` before
        `OutOfBoundsRule.check()` is ever called at all, not merely before
        its result is *used*."""
        referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
        referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

        # Wrap the real, already-constructed OutOfBoundsRule instance's
        # check() with a call counter, in place, so the rest of
        # CustomReferee's internals (the same self._rules list _step()
        # iterates) are completely untouched.
        from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule

        oob_rule = next(r for r in referee._rules if isinstance(r, OutOfBoundsRule))
        call_count = 0
        original_check = oob_rule.check

        def _counting_check(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_check(*args, **kwargs)

        oob_rule.check = _counting_check

        # Ball resting inside the right goal, ball moving into it (GoalRule's
        # trigger condition) — friendly defends left (my_team_is_right=False),
        # so a ball in the right goal is a goal for blue... exact attribution
        # doesn't matter here, only that A goal fires.
        friendly = _robot(0, -3.0, 0.0, True)
        enemy = _robot(0, 3.0, 0.0, False)
        goal_ball = _ball(x=5.0, y=0.0)  # past the right goal line, inside goal width
        goal_ball = Ball(p=Vector3D(5.0, 0.0, 0.0), v=Vector3D(1.0, 0.0, 0.0), a=Vector3D(0, 0, 0))
        frame = _frame(friendly, enemy, ball=goal_ball, my_team_is_right=False, ts=5.0)

        result = referee.step(frame, current_time=5.0)

        assert referee.last_violation is not None
        assert referee.last_violation.rule_name == "goal"
        # _handle_goal transitions straight to a ball-placement command at
        # the centre spot for the conceding team's kickoff, not a bare STOP —
        # what matters here is only that a real command transition happened
        # (proving GoalRule's violation was the one applied), not its
        # specific value.
        assert result.referee_command in (
            RefereeCommand.BALL_PLACEMENT_YELLOW,
            RefereeCommand.BALL_PLACEMENT_BLUE,
        )
        assert call_count == 0, "OutOfBoundsRule.check() must never be called once GoalRule already fired this tick"


class TestScanOrderMechanism:
    def test_earlier_non_stopping_does_not_block_a_later_stopping_violation(self):
        """The one ordering the real rule set's command-gating can't
        currently produce (see module docstring), exercised directly via two
        minimal stub rules registered in that exact order: a non-stopping
        violation found first in the scan must not suppress or pre-empt a
        stopping violation found later in the same tick — the stopping one
        must still win and become `last_violation`/the applied command."""
        referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
        referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

        non_stopping = _StubRule(
            RuleViolation(
                rule_name="stub_non_stopping",
                suggested_command=RefereeCommand.NORMAL_START,
                next_command=None,
                status_message="stub non-stopping",
                is_stopping=False,
            )
        )
        stopping = _StubRule(
            RuleViolation(
                rule_name="stub_stopping",
                suggested_command=RefereeCommand.STOP,
                next_command=RefereeCommand.FORCE_START,
                status_message="stub stopping",
                is_stopping=True,
            )
        )
        # Deliberately ordered non-stopping BEFORE stopping — the direction
        # the real profile's rules can never produce together.
        referee._rules = [non_stopping, stopping]

        friendly = _robot(0, -2.0, 0.0, True)
        enemy = _robot(0, 2.0, 0.0, False)
        result = referee.step(_frame(friendly, enemy, ts=1.0), current_time=1.0)

        assert referee.last_violation is not None
        assert referee.last_violation.rule_name == "stub_stopping"
        assert result.referee_command == RefereeCommand.STOP
        # Both stubs are always "armed" and unconditionally return a
        # violation every tick, so both must have been consulted (no early
        # break before reaching the stopping one, since it comes second).
        assert non_stopping.call_count == 1
        assert stopping.call_count == 1
