from unittest.mock import Mock

from utama_core.custom_referee import CustomReferee
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.rsoccer_simulator.src.ssl import ssl_gym_base
from utama_core.run.strategy_runner import StrategyRunner


def test_strategy_runner_step_once_delegates_to_single_step_loop():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._run_step = Mock()

    runner.step_once()

    runner._run_step.assert_called_once_with()


def test_custom_referee_set_command_accepts_scripted_metadata():
    referee = CustomReferee.from_profile_name("simulation")

    referee.set_command(
        RefereeCommand.BALL_PLACEMENT_YELLOW,
        timestamp=3.0,
        designated_position=(1.0, -0.5),
        next_command=RefereeCommand.NORMAL_START,
        status_message="scripted placement",
    )
    data = referee.step(game_frame=None, current_time=3.0)

    assert data.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW
    assert data.designated_position == (1.0, -0.5)
    assert data.next_command == RefereeCommand.NORMAL_START
    assert data.status_message == "scripted placement"


def test_ssl_base_env_reset_seeds_random_generators(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    monkeypatch.setattr(ssl_gym_base.random, "seed", random_seed)
    monkeypatch.setattr(ssl_gym_base.np.random, "seed", numpy_seed)

    env = ssl_gym_base.SSLBaseEnv.__new__(ssl_gym_base.SSLBaseEnv)
    env.render_mode = None
    env.rsim = Mock()
    env.rsim.get_frame.return_value = object()
    env._get_initial_positions_frame = Mock(return_value=object())
    env._frame_to_observations = Mock(return_value=("obs",))

    result = ssl_gym_base.SSLBaseEnv.reset(env, seed=42)

    random_seed.assert_called_once_with(42)
    numpy_seed.assert_called_once_with(42)
    assert result == (("obs",), {})
