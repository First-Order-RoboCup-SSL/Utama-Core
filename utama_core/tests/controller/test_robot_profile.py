"""Robot profiles: a kickerless robot pushes, a dribblerless one carries gently, a sensorless
one is judged from vision, and only robots the runner drives can have a profile."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utama_core.config.enums import Mode
from utama_core.entities.data.command import RobotCommand
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.team_controller.src.controllers.common.robot_profile import (
    _PUSH_TICKS,
    CARRY_SPEED,
    CARRY_TURN,
    PUSH_SPEED,
    ProfiledRobotController,
    RobotProfile,
)

STOP = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)
KICK = RobotCommand(0.0, 0.0, 0.0, kick=True, chip=False, dribble=False)
TURN = RobotCommand(0.0, 0.0, 1.0, kick=False, chip=False, dribble=True)


class _Recorder:
    def __init__(self):
        self.sent: list[dict] = []

    def add_robot_commands(self, commands, robot_id=None):
        self.sent.append(commands)


def _controller(**profile) -> tuple[ProfiledRobotController, _Recorder]:
    inner = _Recorder()
    return ProfiledRobotController(inner, {1: RobotProfile(**profile)}), inner


def test_a_kickerless_robot_pushes_forward_for_the_whole_push_then_obeys_again():
    controller, inner = _controller(kicker=False)
    controller.add_robot_commands(KICK, 1)
    for _ in range(_PUSH_TICKS - 1):
        controller.add_robot_commands(TURN, 1)  # the tactic's next commands wait for the push
    controller.add_robot_commands(TURN, 1)
    pushes = [c[1] for c in inner.sent[:_PUSH_TICKS]]
    assert all(c.local_forward_vel == PUSH_SPEED and not c.kick and c.angular_vel == 0 for c in pushes)
    assert inner.sent[_PUSH_TICKS][1].angular_vel == 1.0


def test_a_stop_ends_a_push_at_once():
    """HALT and close() send all-zero commands: they must reach the robot mid-push."""
    controller, inner = _controller(kicker=False)
    controller.add_robot_commands(KICK, 1)
    controller.add_robot_commands(STOP, 1)
    controller.add_robot_commands(TURN, 1)
    assert inner.sent[1][1] == STOP
    assert inner.sent[2][1].angular_vel == 1.0


def test_a_dribblerless_robot_never_backs_up_and_carries_at_the_capped_speed():
    controller, inner = _controller(dribbler=False)
    controller.add_robot_commands(RobotCommand(-1.0, 0.0, 0.0, kick=False, chip=False, dribble=True), 1)
    controller.add_robot_commands(RobotCommand(2.0, 0.0, 0.0, kick=False, chip=False, dribble=True), 1)
    backing, fast = inner.sent[0][1], inner.sent[1][1]
    assert backing.local_forward_vel == 0.0 and not backing.dribble
    assert fast.local_forward_vel == pytest.approx(CARRY_SPEED)


def test_a_dribblerless_pivot_keeps_its_shape_when_slowed():
    controller, inner = _controller(dribbler=False)
    controller.add_robot_commands(RobotCommand(0.0, 0.4, 4.0, kick=False, chip=False, dribble=True), 1)
    c = inner.sent[0][1]
    assert c.angular_vel == pytest.approx(CARRY_TURN)
    assert c.local_left_vel / c.angular_vel == pytest.approx(0.4 / 4.0)


def test_a_dribblerless_robot_not_dribbling_is_untouched():
    controller, inner = _controller(dribbler=False)
    fast = RobotCommand(2.0, -1.0, 3.0, kick=False, chip=False, dribble=False)
    controller.add_robot_commands({1: fast, 2: KICK})
    assert inner.sent[0] == {1: fast, 2: KICK}  # robot 2 has no profile


def _runner(mode: Mode, yellow_mapping: dict, **yellow_profiles) -> SimpleNamespace:
    runner = SimpleNamespace(
        mode=mode,
        my_team_is_yellow=True,
        exp_friendly=2,
        exp_enemy=1,
        yellow_vision_to_cmd_mapping=yellow_mapping,
        blue_vision_to_cmd_mapping={},
        yellow_robot_profiles={int(k[1:]): v for k, v in yellow_profiles.items()},  # "r3" -> robot 3
        blue_robot_profiles={},
    )
    runner._driven_ids = lambda is_yellow: StrategyRunner._driven_ids(runner, is_yellow)
    return runner


def test_a_profile_for_a_robot_that_is_not_driven_is_refused():
    """A typo in a test-day config must stop the run, not silently leave a robot kicking air."""
    runner = _runner(Mode.REAL, {4: 0}, r3=RobotProfile(kicker=False))
    with pytest.raises(ValueError, match=r"\[3\]"):
        StrategyRunner._check_profiles_are_driven(runner, is_yellow=True)


def test_the_controller_is_only_wrapped_when_a_kicker_or_dribbler_is_missing():
    runner = _runner(Mode.RSIM, {})
    inner = object()
    assert StrategyRunner._with_profiles(runner, inner, {1: RobotProfile(ir=False)}) is inner
    assert isinstance(
        StrategyRunner._with_profiles(runner, inner, {1: RobotProfile(dribbler=False)}), ProfiledRobotController
    )


def test_a_robot_without_a_ball_sensor_is_left_out_of_the_trusted_ones():
    runner = _runner(Mode.REAL, {4: 0, 6: 1}, r6=RobotProfile(ir=False))
    assert StrategyRunner._trusted_ir(runner, True, None) == frozenset({4})


def test_with_every_ball_sensor_working_the_trusted_set_passes_through():
    runner = _runner(Mode.RSIM, {}, r0=RobotProfile())
    assert StrategyRunner._trusted_ir(runner, True, None) is None  # None: trust all
    assert StrategyRunner._trusted_ir(runner, True, frozenset({1})) == frozenset({1})


def test_ball_sensors_given_both_ways_are_refused():
    runner = _runner(Mode.RSIM, {}, r0=RobotProfile(ir=False))
    with pytest.raises(ValueError, match="not both"):
        StrategyRunner._trusted_ir(runner, True, frozenset({1}))
