import json
import subprocess

import numpy as np


class RSimSubprocessWrapper:
    def __init__(self, sim_type, n_blue, n_yellow, field_type, time_step_ms):
        self.proc = subprocess.Popen(
            [
                "pixi",
                "run",
                "--environment",
                "robosim",
                "--",
                "python",
                "./rsoccer_simulator/src/Simulators/robosim/robosim_subprocess.py",
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
