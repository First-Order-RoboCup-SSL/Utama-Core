import numpy as np
import pytest

from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    RSimSubprocessWrapper,
)

N_PER_TEAM = 6


def _wrapper() -> RSimSubprocessWrapper:
    return RSimSubprocessWrapper("SSL", N_PER_TEAM, N_PER_TEAM, 2, 16)


@pytest.fixture
def two_sims():
    a, b = _wrapper(), _wrapper()
    yield a, b
    a.close()
    b.close()


def test_binary_step_matches_json_step_bit_for_bit(two_sims):
    """The raw-float64 step frame must hand the sim exactly the commands, and
    the runner exactly the state, that the old one-JSON-line-per-tick protocol
    did: two identical sims, one stepped each way with the same commands (full
    precision floats, kicks and dribbler included), stay bit-identical."""
    binary_sim, json_sim = two_sims
    rng = np.random.default_rng(0)
    for tick in range(150):
        cmds = np.zeros((2 * N_PER_TEAM, 8))
        cmds[:, 1:4] = rng.uniform(-2.0, 2.0, (2 * N_PER_TEAM, 3))
        cmds[:, 5] = rng.uniform(0.0, 5.0, 2 * N_PER_TEAM) * (rng.random(2 * N_PER_TEAM) < 0.1)
        cmds[:, 7] = rng.random(2 * N_PER_TEAM) < 0.5
        binary_state = binary_sim.step(cmds)
        json_state = np.array(json_sim._request({"commands": cmds.tolist()})["state"])
        assert binary_state.dtype == np.float64
        assert np.array_equal(binary_state, json_state, equal_nan=True), f"diverged at tick {tick}"


def test_json_commands_still_work_between_binary_steps():
    sim = _wrapper()
    try:
        params = sim.get_field_params()
        assert params["length"] > 0
        sim.step(np.zeros((2 * N_PER_TEAM, 8)))
        sim.reset(
            np.array([0.5, 0.0, 0.0, 0.0]),
            np.array([[-1.0 - i, 0.0, 0.0] for i in range(N_PER_TEAM)]),
            np.array([[1.0 + i, 0.0, 0.0] for i in range(N_PER_TEAM)]),
        )
        state = sim.get_state()
        assert state[0] == pytest.approx(0.5)
        assert len(sim.step(np.zeros((2 * N_PER_TEAM, 8)))) == len(state)
    finally:
        sim.close()


def test_subprocess_error_is_raised_not_swallowed():
    sim = _wrapper()
    try:
        with pytest.raises(RuntimeError, match="robosim subprocess error"):
            sim._request({"no_such_command": True})
    finally:
        sim.close()
