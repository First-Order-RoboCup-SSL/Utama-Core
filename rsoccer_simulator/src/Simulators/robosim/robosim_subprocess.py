"""
This script runs inside Python 3.10 (rc-robosim environment).
It receives JSON commands via stdin and returns simulator state via stdout.
"""

import sys
import json
import numpy as np
import robosim
from rsoccer_simulator.src.Entities import FrameVSS, FrameSSL, Field


# Example: simple wrapper class
class SubprocessRSim:
    def __init__(self, sim_type, n_blue, n_yellow, field_type, time_step_ms):
        self.n_blue = n_blue
        self.n_yellow = n_yellow
        blue_robots_pos = [[-0.2 * i, 0, 0] for i in range(1, self.n_blue + 1)]
        yellow_robots_pos = [[0.2 * i, 0, 0] for i in range(1, self.n_yellow + 1)]
        if sim_type == "VSS":
            self.sim = robosim.VSS(
                field_type,
                n_blue,
                n_yellow,
                time_step_ms,
                [0, 0, 0, 0],
                blue_robots_pos,
                yellow_robots_pos,
            )
        else:
            self.sim = robosim.SSL(
                field_type,
                n_blue,
                n_yellow,
                time_step_ms,
                [0, 0, 0, 0],
                blue_robots_pos,
                yellow_robots_pos,
            )

    def step(self, commands):
        # commands is a numpy array serialized as a list
        arr = np.array(commands)
        self.sim.step(arr)
        # return state as list
        return self.sim.get_state()

    def get_state(self):
        """Return current simulator state without advancing it."""
        return self.sim.get_state()

    def reset(self, ball_pos, blue_robots_pos, yellow_robots_pos):
        self.sim.reset(
            np.array(ball_pos),
            np.array(blue_robots_pos),
            np.array(yellow_robots_pos),
        )

    def get_field_params(self):
        return self.sim.get_field_params()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_type", choices=["VSS", "SSL"], required=True)
    parser.add_argument("--n_blue", type=int, required=True)
    parser.add_argument("--n_yellow", type=int, required=True)
    parser.add_argument("--field_type", type=int, required=True)
    parser.add_argument("--time_step_ms", type=int, required=True)
    args = parser.parse_args()

    sim = SubprocessRSim(
        args.sim_type, args.n_blue, args.n_yellow, args.field_type, args.time_step_ms
    )

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                cmd = json.loads(line)
                if "commands" in cmd:
                    state = sim.step(cmd["commands"])
                    print(json.dumps({"state": state}))
                elif "reset" in cmd:
                    r = cmd["reset"]
                    sim.reset(
                        r["ball_pos"], r["blue_robots_pos"], r["yellow_robots_pos"]
                    )
                    print(json.dumps({"ack": True}))
                elif "get_field_params" in cmd:
                    fp = sim.get_field_params()
                    print(json.dumps({"field_params": fp}))
                elif "get_state" in cmd:
                    state = sim.get_state()
                    print(json.dumps({"state": state}))
                else:
                    print(json.dumps({"error": "unknown command"}))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
