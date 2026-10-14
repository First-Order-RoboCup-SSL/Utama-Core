"""Parent side of the rsim subprocess protocol (`robosim_subprocess.py`).

Requests: `S<uint32 rows><uint32 cols><rows*cols float64>` steps the sim; `J<json
line>` is any other command. Replies: `<kind><uint32 length><payload>`, kind
`B` (a float64 state vector) or `J` (a JSON object). The per-tick step used to
be a JSON line each way; formatting and parsing ~140 floats per tick in two
interpreters was pure overhead on top of the physics, and raw float64 carries
exactly the same values (JSON's float repr round-trips exactly too, so the sim
sees bit-identical inputs and the runner bit-identical states either way).
"""

import atexit
import json
import logging
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Sim reuse (opt-in, `enable_sim_reuse`): a process that plays many short scenarios keeps its
# sim subprocesses, since starting one (`pixi run`, imports, world build) costs about as much
# CPU as a second of play. Idle sims wait here by their constructor arguments; `acquire`
# hands one out with a new native world, so nothing of the previous play is left in it.
_reuse_sims = False
_idle_sims: dict[tuple, list["RSimSubprocessWrapper"]] = {}


def enable_sim_reuse() -> None:
    """From now on, `close()` on a sim from `acquire` parks its subprocess for the next `acquire`
    (closed for good at interpreter exit) instead of terminating it."""
    global _reuse_sims
    if not _reuse_sims:
        _reuse_sims = True
        atexit.register(_close_idle_sims)


def _close_idle_sims() -> None:
    for sims in _idle_sims.values():
        while sims:
            sims.pop()._terminate()


class RSimSubprocessWrapper:
    @classmethod
    def acquire(cls, sim_type, n_blue, n_yellow, field_type, time_step_ms) -> "RSimSubprocessWrapper":
        """A sim in the state a new one would be in: a parked subprocess given a new native world
        when reuse is on and one fits, else a new subprocess."""
        key = (sim_type, n_blue, n_yellow, field_type, time_step_ms)
        idle = _idle_sims.get(key)
        while idle:
            sim = idle.pop()
            try:
                acked = sim._request({"new_world": True}).get("ack")
            except Exception:  # a dead or desynchronised subprocess is not worth debugging: start afresh
                acked = False
            if not acked:
                sim._terminate()
                continue
            sim._released = False
            return sim
        return cls(*key)

    def __init__(self, sim_type, n_blue, n_yellow, field_type, time_step_ms):
        self._key = (sim_type, n_blue, n_yellow, field_type, time_step_ms)
        self._released = False
        self._broken = False  # a request failed part-way: the pipe may be out of step, so never reuse it
        script_path = (Path(__file__).parent / "robosim_subprocess.py").resolve()
        env = os.environ.copy()
        cmake_policy_flag = "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        # Ensure rc-robosim's scikit-build uses a compatible CMake policy level.
        env["CMAKE_ARGS"] = f"{env.get('CMAKE_ARGS', '')} {cmake_policy_flag}".strip()
        env["SKBUILD_CMAKE_ARGS"] = f"{env.get('SKBUILD_CMAKE_ARGS', '')} {cmake_policy_flag}".strip()
        project_root = Path(__file__).resolve().parents[5]
        robosim_env = project_root / ".pixi" / "envs" / "robosim"
        include_dir = robosim_env / "include"
        lib_dir = robosim_env / "lib"
        prefix = str(robosim_env)
        env["CMAKE_PREFIX_PATH"] = f"{prefix}:{env.get('CMAKE_PREFIX_PATH', '')}".strip(":")
        env["CMAKE_LIBRARY_PATH"] = f"{lib_dir}:{env.get('CMAKE_LIBRARY_PATH', '')}".strip(":")
        env["CMAKE_INCLUDE_PATH"] = f"{include_dir}:{env.get('CMAKE_INCLUDE_PATH', '')}".strip(":")
        self.proc = subprocess.Popen(
            [
                "pixi",
                "run",
                "--environment",
                "robosim",
                "--",
                "python",
                str(script_path),
                "--sim_type",
                sim_type,
                "--n_blue",
                str(n_blue),
                "--n_yellow",
                str(n_yellow),
                "--field_type",
                str(field_type),
                "--time_step_ms",
                str(time_step_ms),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=project_root,
        )

    def _read_exact(self, n: int) -> bytes:
        data = self.proc.stdout.read(n)
        if len(data) != n:
            code = self.proc.poll()
            raise RuntimeError(
                f"robosim subprocess closed its stdout while awaiting a response (exit code {code}, "
                f"got {len(data)} of {n} bytes)"
            )
        return data

    def _read_reply(self):
        """One reply frame: a float64 array for `B`, the decoded dict for `J`.

        No tolerance for stray output is needed here (the JSON-line reader it
        replaced skipped up to 10 non-JSON lines): `robosim_subprocess.py`
        points fd 1 at devnull before importing `robosim`, so rc-robosim's
        native "turnover ..." diagnostics never reach this pipe. An unexpected
        kind byte means the framing itself is broken, so fail loudly.
        """
        header = self._read_exact(5)
        kind = header[:1]
        (length,) = struct.unpack("<I", header[1:])
        body = self._read_exact(length)
        if kind == b"B":
            return np.frombuffer(body, dtype="<f8").astype(np.float64)
        if kind == b"J":
            resp = json.loads(body)
            if "error" in resp:
                raise RuntimeError(f"robosim subprocess error: {resp['error']}")
            return resp
        raise RuntimeError(f"robosim subprocess sent an unknown reply frame kind {kind!r}")

    def _exchange(self, request: bytes):
        try:
            self.proc.stdin.write(request)
            self.proc.stdin.flush()
            return self._read_reply()
        except BaseException:
            self._broken = True
            raise

    def _request(self, payload: dict) -> dict:
        return self._exchange(b"J" + json.dumps(payload).encode() + b"\n")

    def step(self, commands: np.ndarray) -> np.ndarray:
        commands = np.ascontiguousarray(commands, dtype="<f8")
        rows, cols = commands.shape
        return self._exchange(b"S" + struct.pack("<II", rows, cols) + commands.tobytes())

    def reset(self, ball_pos, blue_robots_pos, yellow_robots_pos):
        self._request(
            {
                "reset": {
                    "ball_pos": ball_pos.tolist(),
                    "blue_robots_pos": blue_robots_pos.tolist(),
                    "yellow_robots_pos": yellow_robots_pos.tolist(),
                }
            }
        )

    def get_field_params(self):
        return self._request({"get_field_params": True})["field_params"]

    def get_state(self):
        return self._request({"get_state": True})["state"]

    def close(self):
        """Done with this sim: parked for the next `acquire` when reuse is on, else terminated."""
        if self._released:
            return
        if _reuse_sims and not self._broken and self.proc.poll() is None:
            self._released = True
            _idle_sims.setdefault(self._key, []).append(self)
            return
        self._terminate()

    def _terminate(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1)
        except Exception as e:
            import traceback

            logger.error(f"Error while terminating RSim subprocess: {e}")
            traceback.print_exc()
        finally:
            logger.info("RsimSubprocessWrapper cleanup finished.")
