"""Tests for error handling and safety mechanisms in strategies (REAL mode only)."""

from unittest.mock import MagicMock, patch

import pytest

from utama_core.config.enums import Mode


@pytest.fixture
def mock_runner():
    """Create a mock StrategyRunner with REAL mode for testing."""
    from utama_core.run.strategy_runner import StrategyRunner

    with patch.object(StrategyRunner, "__init__", lambda self: None):
        runner = StrategyRunner()
        runner.mode = Mode.REAL
        runner.logger = MagicMock()
        runner.profiler = None
        runner.replay_writer = None
        runner.rsim_env = None
        runner._fps_live = None
        runner._stop_event = MagicMock()
        runner._stop_event.is_set.return_value = False

        # Opp side
        runner.opp = None

        # ---- NEW STRUCTURE ----
        runner.my = MagicMock()
        runner.my.game = MagicMock()
        runner.my.game.friendly_robots = {0: MagicMock(), 1: MagicMock()}
        runner.my.strategy = MagicMock()
        runner.my.strategy.robot_controller = MagicMock()

        yield runner


class TestStopRobotsOnClose:
    """Tests for _stop_robots behavior when close() is called."""

    def test_stop_commands_sent_in_real_mode(self, mock_runner):
        mock_runner.close(stop_command_mult=5)

        controller = mock_runner.my.strategy.robot_controller
        assert controller.add_robot_commands.call_count == 5
        assert controller.send_robot_commands.call_count == 5

    def test_stop_commands_have_zero_velocity(self, mock_runner):
        mock_runner.my.game.friendly_robots = {
            0: MagicMock(),
            1: MagicMock(),
            2: MagicMock(),
        }

        mock_runner._stop_robots(stop_command_mult=1)

        controller = mock_runner.my.strategy.robot_controller
        commands_dict = controller.add_robot_commands.call_args[0][0]

        for robot_id, cmd in commands_dict.items():
            assert cmd.local_forward_vel == 0
            assert cmd.local_left_vel == 0
            assert cmd.angular_vel == 0
            assert not cmd.kick
            assert not cmd.chip
            assert not cmd.dribble

    def test_stop_not_called_in_rsim_mode(self, mock_runner):
        mock_runner.mode = Mode.RSIM
        mock_runner.rsim_env = MagicMock()
        mock_runner._stop_robots = MagicMock()

        mock_runner.close()

        mock_runner._stop_robots.assert_not_called()


class TestStopRobotsOnError:
    """Tests for stop behavior when errors occur during execution."""

    def test_stop_on_runtime_exception(self, mock_runner):
        call_count = {"value": 0}

        def failing_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                raise RuntimeError("Test exception")

        mock_runner._run_step = failing_run_step

        with pytest.raises(RuntimeError, match="Test exception"):
            mock_runner.run()

        controller = mock_runner.my.strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1
        assert controller.send_robot_commands.call_count >= 1

    def test_stop_on_vision_loss(self, mock_runner):
        call_count = {"value": 0}

        def vision_loss_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                raise KeyError("No vision data - camera disconnected")

        mock_runner._run_step = vision_loss_run_step

        with pytest.raises(KeyError, match="No vision data"):
            mock_runner.run()

        controller = mock_runner.my.strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1
        assert controller.send_robot_commands.call_count >= 1

    def test_stop_on_stop_event_signal(self, mock_runner):
        call_count = {"value": 0}

        def signaled_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                mock_runner._stop_event.is_set.return_value = True

        mock_runner._run_step = signaled_run_step

        mock_runner.run()

        controller = mock_runner.my.strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
