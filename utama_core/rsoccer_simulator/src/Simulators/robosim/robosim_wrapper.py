import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class RSimSubprocessWrapper:
    def __init__(self, sim_type, n_blue, n_yellow, field_type, time_step_ms):
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
            text=True,
            bufsize=1,
            env=env,
            cwd=project_root,
        )

    _MAX_NON_JSON_LINES = 10  # generous vs. one stray diagnostic; see docstring below

    def _read_json_line(self) -> dict:
        # rc-robosim's native layer occasionally writes plain-text diagnostics
        # (e.g. "turnover 0.862573 robot x: ... ball y: ...") straight to its
        # own stdout, which shares this pipe with our JSON RPC protocol --
        # confirmed live, 2026-09-01: a `full_match_tournament.py` match
        # deterministically hit this at the exact same tick every run, and
        # the subprocess was still alive (`proc.poll() is None`) when it
        # happened, so this was never a subprocess crash despite raising
        # JSONDecodeError one line up the stack. Skip any line that isn't
        # valid JSON instead of letting it blow up the caller.
        #
        # IMPORTANT: an unbounded skip loop is not safe here. Verified live
        # that at least one "turnover" diagnostic came out *instead of* that
        # tick's JSON state reply, not merely ahead of it: after discarding
        # it, `readline()` blocked forever waiting on a reply that was never
        # coming, while the subprocess sat idle (`S` state, 0 bytes buffered
        # on the pipe) waiting for its *next* command -- a silent, one-sided
        # deadlock, strictly worse than the JSONDecodeError this replaced
        # (that at least surfaced loudly). Cap the number of stray lines
        # tolerated per call and fail loudly past that instead of hanging.
        for _ in range(self._MAX_NON_JSON_LINES):
            if self.proc.poll() is not None:
                raise RuntimeError(f"robosim subprocess exited (code {self.proc.returncode}) while awaiting a response")
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("robosim subprocess closed its stdout while awaiting a response")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Discarding non-JSON line from robosim subprocess stdout: %r", line)
        raise RuntimeError(
            f"robosim subprocess sent {self._MAX_NON_JSON_LINES} consecutive non-JSON lines "
            "without a valid reply -- treating this as a missing response, not more diagnostics to skip"
        )

    def step(self, commands: np.ndarray):
        # Serialize commands as JSON and send to subprocess
        data = json.dumps({"commands": commands.tolist()})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()

        # Read simulator state back
        state = self._read_json_line()["state"]
        return np.array(state)

    def reset(self, ball_pos, blue_robots_pos, yellow_robots_pos):
        data = json.dumps(
            {
                "reset": {
                    "ball_pos": ball_pos.tolist(),
                    "blue_robots_pos": blue_robots_pos.tolist(),
                    "yellow_robots_pos": yellow_robots_pos.tolist(),
                }
            }
        )
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()
        # Optionally read acknowledgement
        self._read_json_line()

    def get_field_params(self):
        data = json.dumps({"get_field_params": True})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()
        resp = self._read_json_line()
        return resp["field_params"]

    def get_state(self):
        data = json.dumps({"get_state": True})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()
        resp = self._read_json_line()
        return resp["state"]

    def close(self):
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
