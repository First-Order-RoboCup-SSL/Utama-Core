"""Check the velocity sent on the real radio, including diagonal commands."""

import math
import struct
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from utama_core.config.robot_params import REAL_PARAMS
from utama_core.config.settings import (
    KICKER_COOLDOWN_TIMESTEPS,
    KICKER_PERSIST_TIMESTEPS,
)
from utama_core.entities.data.command import RobotCommand
from utama_core.team_controller.src.controllers.real import (
    real_robot_controller as real_module,
)
from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    RealRobotController,
)


@pytest.mark.parametrize("forward,left", [(2.0, 0.0), (0.8, 0.8), (-1.0, 1.0), (100.0, -1.0)])
def test_radio_limits_total_translation_speed_and_preserves_direction(forward, left):
    serial = MagicMock()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={3: 5}, serial_port=serial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        controller.add_robot_commands(RobotCommand(forward, left, 10.0, False, False, False), 3)
    controller.send_robot_commands()
    packet = bytes(serial.write.call_args.args[0])
    assert packet[0] == 0xAA and packet[1] == 5 and packet[-1] == 0x55
    vx, vy, omega = struct.unpack_from("<eee", packet, 2)
    assert math.hypot(vx, vy) <= REAL_PARAMS.MAX_VEL
    assert abs(omega) <= REAL_PARAMS.MAX_ANGULAR_VEL
    scale = REAL_PARAMS.MAX_VEL / math.hypot(forward, left)
    assert (vx, vy) == pytest.approx((forward * scale, left * scale), abs=0.001)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_radio_replaces_nonfinite_velocities_with_zero(value):
    serial = MagicMock()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={0: 0}, serial_port=serial)
    controller.add_robot_commands(RobotCommand(value, value, value, False, False, False), 0)
    controller.send_robot_commands()
    assert struct.unpack_from("<eee", bytes(serial.write.call_args.args[0]), 2) == (0.0, 0.0, 0.0)


def test_binary16_rounding_cannot_push_a_command_over_the_speed_cap():
    controller = RealRobotController(True, 1, serial_port=MagicMock())
    for angle in np.linspace(-math.pi, math.pi, 361):
        command = RobotCommand(math.cos(angle), math.sin(angle), 0.0, False, False, False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            encoded = controller._convert_float16_command(0, command)
        forward = float(encoded.local_forward_vel.view(np.float16))
        left = float(encoded.local_left_vel.view(np.float16))
        assert math.hypot(forward, left) <= REAL_PARAMS.MAX_VEL


def test_repeated_kicks_obey_the_radio_persistence_and_cooldown():
    serial = MagicMock()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={0: 0}, serial_port=serial)
    flags = []
    for _ in range(KICKER_COOLDOWN_TIMESTEPS + 1):
        controller.add_robot_commands(RobotCommand(0, 0, 0, True, False, False), 0)
        controller.send_robot_commands()
        flags.append(bytes(serial.write.call_args.args[0])[10])
    assert flags[:KICKER_PERSIST_TIMESTEPS] == [0xF0] * KICKER_PERSIST_TIMESTEPS
    assert flags[KICKER_PERSIST_TIMESTEPS:-1] == [0] * (KICKER_COOLDOWN_TIMESTEPS - KICKER_PERSIST_TIMESTEPS)
    assert flags[-1] == 0xF0


def test_dribbler_thermal_cutoff_is_applied_to_the_transmitted_packet(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(real_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    serial = MagicMock()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={0: 0}, serial_port=serial)

    def send_dribble():
        controller.add_robot_commands(RobotCommand(0, 0, 0, False, False, True), 0)
        controller.send_robot_commands()
        return int.from_bytes(bytes(serial.write.call_args.args[0])[8:10], "big")

    assert send_dribble() != 0
    clock[0] = real_module.DRIBBLER_MAX_ON_SECONDS
    assert send_dribble() != 0  # the elapsed interval fills the bucket; the next command is throttled
    with pytest.warns(UserWarning, match="thermal limit"):
        assert send_dribble() == 0
    clock[0] += (
        real_module.DRIBBLER_MAX_ON_SECONDS - real_module.DRIBBLER_RESUME_SECONDS
    ) / real_module.DRIBBLER_DRAIN_RATE
    assert send_dribble() == 0  # this interval drains the bucket and releases the throttle
    assert send_dribble() != 0
