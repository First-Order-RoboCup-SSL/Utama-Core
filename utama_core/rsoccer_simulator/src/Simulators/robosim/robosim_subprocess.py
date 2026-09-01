"""This script runs inside Python 3.10 (rc-robosim environment).

It receives JSON commands via stdin and returns simulator state via stdout.
"""

import json
import os
import sys

import numpy as np
import robosim

# rc-robosim's native (C++) layer occasionally writes plain-text diagnostics
# (e.g. "turnover 0.86 robot x: ... ball y: ...") straight to the process's
# real stdout fd via printf/std::cout -- bypassing `sys.stdout` entirely, so
# nothing on the Python side can intercept or filter it. That text lands on
# the exact same pipe as this protocol's JSON replies. Confirmed live,
# 2026-09-01: one such line came out *in place of* a tick's JSON state
# reply (not just interleaved before it), which either raises a
# JSONDecodeError in `robosim_wrapper.py` or, if the caller instead loops
# skipping non-JSON lines, can deadlock forever waiting for a reply that
# was silently replaced rather than delayed.
#
# Fix at the source: duplicate the original stdout fd to a private fd our
# own `print()`s use, then repoint the real fd 1 at devnull *before*
# `robosim` (the native extension) is even imported/constructs anything --
# any native write to fd 1 lands in devnull, and our own protocol replies
# go out `_PROTOCOL_OUT` on the untouched duplicate, so the pipe our parent
# reads from only ever sees valid JSON.
_PROTOCOL_FD = os.dup(1)
_PROTOCOL_OUT = os.fdopen(_PROTOCOL_FD, "w", buffering=1)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)
os.close(_devnull_fd)


def _emit(payload: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_OUT.flush()


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

    sim = SubprocessRSim(args.sim_type, args.n_blue, args.n_yellow, args.field_type, args.time_step_ms)

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                cmd = json.loads(line)
                if "commands" in cmd:
                    state = sim.step(cmd["commands"])
                    _emit({"state": state})
                elif "reset" in cmd:
                    r = cmd["reset"]
                    sim.reset(r["ball_pos"], r["blue_robots_pos"], r["yellow_robots_pos"])
                    _emit({"ack": True})
                elif "get_field_params" in cmd:
                    fp = sim.get_field_params()
                    _emit({"field_params": fp})
                elif "get_state" in cmd:
                    state = sim.get_state()
                    _emit({"state": state})
                else:
                    _emit({"error": "unknown command"})
            except Exception as e:
                _emit({"error": str(e)})
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
