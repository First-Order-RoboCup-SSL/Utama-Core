import json
import os
import subprocess
from pathlib import Path

import numpy as np


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
        )

    def step(self, commands: np.ndarray):
        # Serialize commands as JSON and send to subprocess
        data = json.dumps({"commands": commands.tolist()})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()

        # Read simulator state back
        line = self.proc.stdout.readline()
        state = json.loads(line)["state"]
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
        self.proc.stdout.readline()

    def get_field_params(self):
        data = json.dumps({"get_field_params": True})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        return resp["field_params"]

    def get_state(self):
        data = json.dumps({"get_state": True})
        self.proc.stdin.write(data + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        return resp["state"]

    def close(self):
        self.proc.terminate()
        self.proc.wait()
