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
        runner.opp_game = None
        runner.opp_strategy = None

        # Mock game with robots
        runner.my_game = MagicMock()
        runner.my_game.friendly_robots = {0: MagicMock(), 1: MagicMock()}

        # Mock strategy and robot controller
        runner.my_strategy = MagicMock()
        runner.my_strategy.robot_controller = MagicMock()

        yield runner


class TestStopRobotsOnClose:
    """Tests for _stop_robots behavior when close() is called."""

    def test_stop_commands_sent_in_real_mode(self, mock_runner):
        """Verify stop commands are sent when close() is invoked in REAL mode."""
        mock_runner.close(stop_command_mult=5)

        controller = mock_runner.my_strategy.robot_controller
        assert controller.add_robot_commands.call_count == 5
        assert controller.send_robot_commands.call_count == 5

    def test_stop_commands_have_zero_velocity(self, mock_runner):
        """Verify stop commands have all zero velocities and disabled actuators."""
        mock_runner.my_game.friendly_robots = {
            0: MagicMock(),
            1: MagicMock(),
            2: MagicMock(),
        }
        mock_runner._stop_robots(stop_command_mult=1)

        controller = mock_runner.my_strategy.robot_controller
        commands_dict = controller.add_robot_commands.call_args[0][0]

        for robot_id, cmd in commands_dict.items():
            assert cmd.local_forward_vel == 0, f"Robot {robot_id}: non-zero forward vel"
            assert cmd.local_left_vel == 0, f"Robot {robot_id}: non-zero left vel"
            assert cmd.angular_vel == 0, f"Robot {robot_id}: non-zero angular vel"
            assert not cmd.kick, f"Robot {robot_id}: kick enabled"
            assert not cmd.chip, f"Robot {robot_id}: chip enabled"
            assert not cmd.dribble, f"Robot {robot_id}: dribble enabled"

    def test_stop_not_called_in_rsim_mode(self, mock_runner):
        """Verify stop commands are NOT sent in RSIM mode."""
        mock_runner.mode = Mode.RSIM
        mock_runner.rsim_env = MagicMock()
        mock_runner._stop_robots = MagicMock()

        mock_runner.close()

        mock_runner._stop_robots.assert_not_called()


class TestStopRobotsOnError:
    """Tests for stop behavior when errors occur during execution."""

    def test_stop_on_runtime_exception(self, mock_runner):
        """Verify stop commands sent when exception occurs during run()."""
        call_count = {"value": 0}

        def failing_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                raise RuntimeError("Test exception")

        mock_runner._run_step = failing_run_step

        with pytest.raises(RuntimeError, match="Test exception"):
            mock_runner.run()

        controller = mock_runner.my_strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1
        assert controller.send_robot_commands.call_count >= 1

    def test_stop_on_vision_loss(self, mock_runner):
        """Verify stop commands sent when vision data is lost."""
        call_count = {"value": 0}

        def vision_loss_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                raise KeyError("No vision data - camera disconnected")

        mock_runner._run_step = vision_loss_run_step

        with pytest.raises(KeyError, match="No vision data"):
            mock_runner.run()

        controller = mock_runner.my_strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1, "Stop commands not sent after vision loss"
        assert controller.send_robot_commands.call_count >= 1, "Stop commands not transmitted"

    def test_stop_on_stop_event_signal(self, mock_runner):
        """Verify stop commands sent when stop event is signaled (e.g., SIGINT handler)."""
        call_count = {"value": 0}

        def signaled_run_step():
            call_count["value"] += 1
            if call_count["value"] >= 3:
                # Simulate SIGINT being handled - sets stop_event
                mock_runner._stop_event.is_set.return_value = True

        mock_runner._run_step = signaled_run_step

        # Should exit gracefully when stop_event is set
        mock_runner.run()

        controller = mock_runner.my_strategy.robot_controller
        assert controller.add_robot_commands.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
