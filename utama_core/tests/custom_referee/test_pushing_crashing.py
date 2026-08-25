"""Unit tests for PushingRule (SSL rulebook §8.4.1) and CrashingRule (§8.4.2)."""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.crashing_rule import CrashingRule
from utama_core.custom_referee.rules.pushing_rule import PushingRule
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)

# Robots placed exactly at contact distance (2*ROBOT_RADIUS=0.18) along the
# x-axis: friendly at x=0, enemy at x=0.18, so "closing" for the friendly is
# +vx and for the enemy is -vx.
_CONTACT_X = 0.18


def _ball(x: float = 5.0, y: float = 5.0) -> Ball:
    # Ball placed away from both robots — these rules don't depend on ball
    # position, only robot-pair contact/velocity.
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
    friendly: Robot | None,
    enemy: Robot | None,
    my_team_is_yellow: bool = True,
    ts: float = 10.0,
) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=False,
        friendly_robots={friendly.id: friendly} if friendly else {},
        enemy_robots={enemy.id: enemy} if enemy else {},
        ball=_ball(),
        referee=None,
    )


# ---------------------------------------------------------------------------
# PushingRule
# ---------------------------------------------------------------------------


class TestPushingRule:
    def test_no_contact_no_violation(self):
        rule = PushingRule(persistence_frames=3)
        friendly = _robot(0, 0.0, 0.0, True, vx=1.0)
        enemy = _robot(0, 5.0, 5.0, False)  # far away — no contact
        frame = _frame(friendly, enemy)
        for _ in range(10):
            assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_sustained_asymmetric_push_friendly_dominant(self):
        rule = PushingRule(min_closing_speed_mps=0.05, similar_force_margin_mps=0.15, persistence_frames=3)
        # Friendly closes fast (+vx toward enemy); enemy still closes a
        # little (both must be "travelling towards the opponent" per the
        # rulebook) but far less, so friendly is the clear pusher.
        friendly = _robot(0, 0.0, 0.0, True, vx=1.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=-0.1)
        frame = _frame(friendly, enemy, my_team_is_yellow=True)

        violations = [rule.check(frame, GEO, RefereeCommand.NORMAL_START) for _ in range(3)]
        assert violations[0] is None
        assert violations[1] is None
        violation = violations[2]
        assert violation is not None
        assert violation.rule_name == "pushing"
        assert violation.is_stopping
        assert violation.offending_teams == (True,)  # friendly (yellow) is the pusher
        assert violation.next_command == RefereeCommand.DIRECT_FREE_BLUE  # non-pusher gets free kick

    def test_sustained_asymmetric_push_enemy_dominant_blue_team(self):
        rule = PushingRule(min_closing_speed_mps=0.05, similar_force_margin_mps=0.15, persistence_frames=3)
        # Enemy closes fast, friendly closes only a little.
        friendly = _robot(0, 0.0, 0.0, True, vx=0.1)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=-1.0)
        frame = _frame(friendly, enemy, my_team_is_yellow=True)  # friendly=yellow, enemy=blue

        for _ in range(2):
            rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.offending_teams == (False,)  # enemy (blue) is the pusher
        assert violation.next_command == RefereeCommand.DIRECT_FREE_YELLOW

    def test_symmetric_push_no_fault(self):
        rule = PushingRule(min_closing_speed_mps=0.05, similar_force_margin_mps=0.15, persistence_frames=3)
        # Both close at the same speed — similar force, no team at fault.
        friendly = _robot(0, 0.0, 0.0, True, vx=1.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=-1.0)
        frame = _frame(friendly, enemy)

        for _ in range(2):
            rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.offending_teams == ()
        assert violation.counts_toward_foul_counter is True  # no offending team to charge regardless
        assert violation.next_command == RefereeCommand.FORCE_START
        assert violation.designated_position is not None

    def test_non_sustained_contact_resets(self):
        rule = PushingRule(min_closing_speed_mps=0.05, similar_force_margin_mps=0.15, persistence_frames=3)
        friendly = _robot(0, 0.0, 0.0, True, vx=1.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame = _frame(friendly, enemy)
        far_enemy = _robot(0, 5.0, 5.0, False)
        no_contact_frame = _frame(friendly, far_enemy)

        rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        # Contact breaks before persistence threshold — counter should reset.
        rule.check(no_contact_frame, GEO, RefereeCommand.NORMAL_START)
        v1 = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        v2 = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v1 is None
        assert v2 is None  # would have fired on the 3rd consecutive tick had it not reset


# ---------------------------------------------------------------------------
# CrashingRule
# ---------------------------------------------------------------------------


class TestCrashingRule:
    def test_no_contact_no_violation(self):
        rule = CrashingRule()
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, 5.0, 5.0, False)
        frame = _frame(friendly, enemy)
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_rising_edge_one_robot_faster(self):
        rule = CrashingRule(fault_speed_threshold_mps=1.5, both_fault_threshold_mps=0.3)
        # Enemy stationary, friendly closing at 2 m/s -> friendly is "faster".
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame = _frame(friendly, enemy, my_team_is_yellow=True)

        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.rule_name == "crashing"
        assert violation.is_stopping is False
        assert violation.offending_teams == (True,)  # friendly=yellow was faster

    def test_no_rising_edge_no_retrigger_within_cooldown(self):
        rule = CrashingRule(fault_speed_threshold_mps=1.5, both_fault_threshold_mps=0.3, retrigger_cooldown_seconds=2.0)
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame_t0 = _frame(friendly, enemy, ts=10.0)
        frame_t1 = _frame(friendly, enemy, ts=11.0)  # still in contact, only 1s later

        v0 = rule.check(frame_t0, GEO, RefereeCommand.NORMAL_START)
        v1 = rule.check(frame_t1, GEO, RefereeCommand.NORMAL_START)
        assert v0 is not None
        assert v1 is None  # sustained contact, cooldown not yet elapsed

    def test_retrigger_after_cooldown_elapsed(self):
        rule = CrashingRule(fault_speed_threshold_mps=1.5, both_fault_threshold_mps=0.3, retrigger_cooldown_seconds=2.0)
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame_t0 = _frame(friendly, enemy, ts=10.0)
        frame_t3 = _frame(friendly, enemy, ts=13.0)  # sustained contact, 3s later

        v0 = rule.check(frame_t0, GEO, RefereeCommand.NORMAL_START)
        v1 = rule.check(frame_t3, GEO, RefereeCommand.NORMAL_START)
        assert v0 is not None
        assert v1 is not None  # cooldown elapsed while still in contact

    def test_matched_speed_both_fault(self):
        rule = CrashingRule(fault_speed_threshold_mps=1.5, both_fault_threshold_mps=0.3)
        # Head-on at matched closing speed -> both foul.
        friendly = _robot(0, 0.0, 0.0, True, vx=1.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=-1.0)
        frame = _frame(friendly, enemy)

        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert set(violation.offending_teams) == {True, False}
        assert violation.is_stopping is False

    def test_gap_between_thresholds_no_foul(self):
        # closing_diff between both_fault_threshold and fault_speed_threshold:
        # neither band applies per rulebook text.
        rule = CrashingRule(fault_speed_threshold_mps=1.5, both_fault_threshold_mps=0.3)
        friendly = _robot(0, 0.0, 0.0, True, vx=0.8)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame = _frame(friendly, enemy)

        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is None

    def test_not_active_play_no_violation(self):
        rule = CrashingRule()
        friendly = _robot(0, 0.0, 0.0, True, vx=2.0)
        enemy = _robot(0, _CONTACT_X, 0.0, False, vx=0.0)
        frame = _frame(friendly, enemy)
        assert rule.check(frame, GEO, RefereeCommand.STOP) is None
