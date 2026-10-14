import os

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


def _descendant_running(pid: int, name: str) -> int:
    """`proc.pid` is the `pixi run` launcher, whose own argv names the script too:
    the script is the deepest process under it with `name` in its argv."""
    found, pending = None, [pid]
    while pending:
        p = pending.pop(0)
        with open(f"/proc/{p}/cmdline", "rb") as f:
            if name.encode() in f.read():
                found = p
        for task in os.listdir(f"/proc/{p}/task"):
            with open(f"/proc/{p}/task/{task}/children") as f:
                pending += [int(c) for c in f.read().split()]
    assert found not in (None, pid), f"no process under {pid} runs {name}"
    return found


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_subprocess_stdout_goes_to_devnull():
    """rc-robosim's C++ layer prints diagnostics straight to fd 1. The binary
    frames can't skip stray bytes the way the old JSON-line reader skipped stray
    lines, so fd 1 in the subprocess must not be the pipe the parent reads."""
    sim = _wrapper()
    try:
        sim.get_field_params()  # the subprocess is up and past its imports
        script = _descendant_running(sim.proc.pid, "robosim_subprocess.py")
        assert os.readlink(f"/proc/{script}/fd/1") == os.devnull
    finally:
        sim.close()


def _play(sim: RSimSubprocessWrapper, seed: int, ticks: int = 200) -> np.ndarray:
    """A short play from a fixed placement: every robot driving hard, dribblers on, kicks fired."""
    sim.reset(
        np.array([0.3, 0.1, 0.0, 0.0]),
        np.array([[-0.5 - 0.4 * i, 0.3 * (i % 3), 0.0] for i in range(N_PER_TEAM)]),
        np.array([[0.5 + 0.4 * i, -0.3 * (i % 3), 3.1] for i in range(N_PER_TEAM)]),
    )
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(ticks):
        cmds = np.zeros((2 * N_PER_TEAM, 8))
        cmds[:, 1:4] = rng.uniform(-2.0, 2.0, (2 * N_PER_TEAM, 3))
        cmds[:, 5] = rng.uniform(0.0, 5.0, 2 * N_PER_TEAM) * (rng.random(2 * N_PER_TEAM) < 0.1)
        cmds[:, 7] = rng.random(2 * N_PER_TEAM) < 0.5
        states.append(sim.step(cmds))
    return np.array(states)


def test_reused_sim_plays_exactly_as_a_fresh_one(monkeypatch):
    """With reuse on, closing a sim parks its subprocess and the next `acquire` hands it back with a
    new native world: a play after a different, messy play (ball kicked away, robots fast and
    turned, dribblers spinning) must be bit-identical to the same play in a fresh subprocess."""
    from utama_core.rsoccer_simulator.src.Simulators.robosim import (
        robosim_wrapper as rw,
    )

    monkeypatch.setattr(rw, "_reuse_sims", True)
    monkeypatch.setattr(rw, "_idle_sims", {})
    key = ("SSL", N_PER_TEAM, N_PER_TEAM, 2, 16)

    fresh = RSimSubprocessWrapper(*key)
    try:
        expected = _play(fresh, seed=7)
    finally:
        fresh._terminate()

    first = RSimSubprocessWrapper.acquire(*key)
    pid = first.proc.pid
    try:
        _play(first, seed=1)
        first.close()  # parked, not terminated
        assert first.proc.poll() is None
        second = RSimSubprocessWrapper.acquire(*key)
        assert second.proc.pid == pid, "the parked subprocess was not reused"
        assert np.array_equal(_play(second, seed=7), expected)
        second.close()
    finally:
        rw._close_idle_sims()
    assert first.proc.poll() is not None


def test_sim_whose_request_failed_is_not_parked_for_reuse(monkeypatch):
    """A request that failed part-way may have left the pipe out of step, so that sim is terminated."""
    from utama_core.rsoccer_simulator.src.Simulators.robosim import (
        robosim_wrapper as rw,
    )

    monkeypatch.setattr(rw, "_reuse_sims", True)
    monkeypatch.setattr(rw, "_idle_sims", {})
    sim = RSimSubprocessWrapper.acquire("SSL", N_PER_TEAM, N_PER_TEAM, 2, 16)
    try:
        with pytest.raises(RuntimeError):
            sim._request({"no_such_command": True})
        sim.close()
        assert sim.proc.poll() is not None
        assert not rw._idle_sims.get(("SSL", N_PER_TEAM, N_PER_TEAM, 2, 16))
    finally:
        rw._close_idle_sims()
        sim._terminate()
