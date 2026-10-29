"""Exercise the RL field-test runner with only camera, radio and dashboard I/O replaced."""

import dataclasses
import hashlib
import importlib
import json
import math
import struct
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tools import striker_duel
from tools.striker_duel import ACTIONS, N_OBS, StrikerTactic
from utama_core.config.enums import Mode
from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.config.robot_params import REAL_PARAMS
from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import strategy_runner as runner_module
from utama_core.run.field_check import VISION_LOST_S
from utama_core.run.field_fence import FencedRobotController
from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    RealRobotController,
)
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot


def test_default_checkpoint_loads_from_another_working_directory(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples"))
    example = importlib.import_module("real_striker")
    monkeypatch.chdir(tmp_path)
    weights = example.DEFAULT_WEIGHTS
    metadata = json.loads(weights.with_suffix(".json").read_text())
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["actions"] == list(ACTIONS)
    params = striker_duel.load_params(weights)
    assert set(params) == set(metadata["network"]["arrays"])
    for name, shape in metadata["network"]["arrays"].items():
        assert list(params[name].shape) == shape
        assert np.isfinite(params[name]).all()
    fake_runner = MagicMock()
    monkeypatch.setattr(example.setup, "build_runner", fake_runner)
    example.build_runner("real")
    tactic = fake_runner.call_args.kwargs["attacker_tactics"]["striker_rl"]()
    action, logp = tactic.choose(np.zeros(N_OBS, dtype=np.float32), np.ones(len(ACTIONS), dtype=bool))
    assert 0 <= action < len(ACTIONS) and np.isfinite(logp)


@pytest.fixture
def live_striker(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples"))
    example = importlib.import_module("real_striker")
    setup = example.setup
    monkeypatch.setattr(setup, "ATTACKER", setup.TestRobot(vision_id=3, firmware_id=5))
    monkeypatch.setattr(setup, "DEFENDER", setup.TestRobot(vision_id=4, firmware_id=6))
    monkeypatch.setattr(setup, "CAMERA_HALF", "right")
    monkeypatch.setattr(setup, "ATTACKER_COLOUR", "yellow")
    monkeypatch.setattr(setup, "RECORD", False)
    monkeypatch.setattr(setup, "attach_dashboard", MagicMock())
    monkeypatch.setattr(setup.referee_view, "attach", MagicMock())
    monkeypatch.setattr(setup.strategy_view, "attach", MagicMock())

    controllers = []

    def radio(**kwargs):
        serial = kwargs.pop("serial_port", None)
        if serial is None:
            serial = MagicMock()
            serial.in_waiting = 0
        controller = RealRobotController(**kwargs, serial_port=serial)
        controller.add_robot_commands = MagicMock(wraps=controller.add_robot_commands)
        controllers.append(controller)
        return controller

    monkeypatch.setattr(runner_module, "RealRobotController", radio)
    monkeypatch.setattr(runner_module.StrategyRunner, "_setup_vision_and_referee", lambda self: ([], deque()))
    game = build_game(
        {3: make_robot(3, is_friendly=True, x=0.0)},
        {4: make_robot(4, is_friendly=False, x=1.35)},
        make_ball(0.11, 0.0),
        field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        my_team_is_right=False,
        ts=100.0,
    )
    frame = game._current_game_standard_frame
    opposite = dataclasses.replace(
        frame,
        friendly_robots=frame.enemy_robots,
        enemy_robots=frame.friendly_robots,
        my_team_is_yellow=False,
        my_team_is_right=True,
    )
    monkeypatch.setattr(runner_module.GameGater, "wait_until_game_valid", lambda *args, **kwargs: (frame, opposite))
    weights = tmp_path / "weights.npz"
    np.savez(
        weights,
        w0=np.zeros((28, 2)),
        b0=np.zeros(2),
        w1=np.zeros((2, 2)),
        b1=np.zeros(2),
        w2=np.zeros((2, 16)),
        b2=np.zeros(16),
    )
    runner = example.build_runner("real", weights)
    try:
        yield runner, controllers
    finally:
        runner.close(stop_command_repeat=1)


def test_rl_uses_the_small_field_configured_robot_ids_and_fenced_controller(live_striker):
    runner, controllers = live_striker
    assert runner.full_field_dims == GREAT_EXHIBITION_FIELD_DIMS
    assert runner.referee_initial_command == RefereeCommand.HALT
    assert runner.sim_controller is None
    assert runner.yellow_vision_to_cmd_mapping == {3: 5}
    assert runner.blue_vision_to_cmd_mapping == {4: 6}
    assert runner.field_check.expected == [("blue", 4), ("yellow", 3)]
    tactic = runner.my.strategy._kernel_strategy._tactics["play"]
    assert isinstance(tactic, StrikerTactic) and tactic.robot_id == 3
    assert isinstance(runner.my.strategy.robot_controller, FencedRobotController)
    # A command towards the invisible half is stopped at the half-way line, below the tactic.
    runner.my.strategy.robot_controller.add_robot_commands({3: RobotCommand(-1, 0, 0, False, False, False)})
    sent = controllers[0].add_robot_commands.call_args.args[0][3]
    assert sent.local_forward_vel == 0
    controllers[0].send_robot_commands()


def test_live_rl_halts_on_lost_vision_and_stays_halted(live_striker):
    runner, _ = live_striker
    runner.field_check.observe_geometry(3000, 2250)
    visible = RawVisionData(
        ts=100.0,
        yellow_robots=[RawRobotData(3, 0.0, 0.0, 0.0, 1.0)],
        blue_robots=[RawRobotData(4, 1.35, 0.0, 0.0, 1.0)],
        balls=[RawBallData(0.11, 0.0, 0.0, 1.0)],
        camera_id=0,
    )
    runner.referee.force_command(RefereeCommand.FORCE_START, 100.0)
    runner._prev_custom_ref_command = RefereeCommand.FORCE_START
    runner._tick_field_check([visible], now=100.0)
    assert runner._auto_halt is None
    runner._tick_field_check([], now=100.0 + VISION_LOST_S + 0.01)
    assert runner._auto_halt is not None
    data = runner.referee.step(runner.my.current_game_frame, 101.0)
    assert data.referee_command == RefereeCommand.HALT
    data = runner.referee.step(runner.my.current_game_frame, 120.0)
    assert data.referee_command == RefereeCommand.HALT


def test_live_rl_planner_uses_real_speed_turn_and_acceleration_limits(live_striker):
    runner, _ = live_striker
    controller = runner.my.strategy.motion_controller
    assert controller.mode == Mode.REAL
    last_velocity, last_turn = Vector2D(0.0, 0.0), 0.0
    velocities, turns = [], []
    for _ in range(60):
        velocity = controller.pid_trans.calculate(Vector2D(5.0, 5.0), Vector2D(0.0, 0.0), 3)
        turn = controller.pid_oren.calculate(math.pi, 0.0, 3)
        assert velocity.mag() <= REAL_PARAMS.MAX_VEL + 1e-12
        assert (velocity - last_velocity).mag() / TIMESTEP <= REAL_PARAMS.MAX_ACCELERATION + 1e-12
        assert abs(turn) <= REAL_PARAMS.MAX_ANGULAR_VEL
        assert abs(turn - last_turn) / TIMESTEP <= REAL_PARAMS.MAX_ANGULAR_ACCELERATION + 1e-12
        velocities.append(velocity.mag())
        turns.append(abs(turn))
        last_velocity, last_turn = velocity, turn
    assert max(velocities) == pytest.approx(REAL_PARAMS.MAX_VEL)
    assert max(turns) == pytest.approx(REAL_PARAMS.MAX_ANGULAR_VEL)


@pytest.mark.parametrize("action", range(len(ACTIONS)), ids=ACTIONS)
def test_every_rl_action_goes_through_the_real_packet_limits(live_striker, monkeypatch, action):
    runner, controllers = live_striker
    tactic = runner.my.strategy._kernel_strategy._tactics["play"]
    # Exercise every action the policy can select, with a ball held on the dribbler.
    tactic.choose = lambda obs, mask: (action, 0.0)
    monkeypatch.setattr(striker_duel, "has_ball", lambda *args, **kwargs: True)
    runner.my.strategy.step()
    packet = bytes(controllers[0].serial_port.write.call_args.args[0])
    assert packet[1] == 5
    forward, left, turn = struct.unpack_from("<eee", packet, 2)
    assert math.hypot(forward, left) <= REAL_PARAMS.MAX_VEL
    assert abs(turn) <= REAL_PARAMS.MAX_ANGULAR_VEL
    assert len(tactic.trajectory) == 1


def test_a_tactic_cannot_send_an_overspeed_command_to_the_radio(live_striker, monkeypatch):
    runner, controllers = live_striker
    tactic = runner.my.strategy._kernel_strategy._tactics["play"]
    tactic.choose = lambda obs, mask: (0, 0.0)
    monkeypatch.setattr(striker_duel, "has_ball", lambda *args, **kwargs: True)
    monkeypatch.setattr(tactic, "_execute", lambda *args: RobotCommand(2.0, 2.0, 20.0, False, False, True))
    with pytest.warns(UserWarning):
        runner.my.strategy.step()
    packet = bytes(controllers[0].serial_port.write.call_args.args[0])
    forward, left, turn = struct.unpack_from("<eee", packet, 2)
    assert math.hypot(forward, left) <= REAL_PARAMS.MAX_VEL
    assert abs(turn) == REAL_PARAMS.MAX_ANGULAR_VEL
