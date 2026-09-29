"""This script runs inside Python 3.10 (rc-robosim environment).

It receives commands via stdin and returns simulator state via stdout, framed
as described in `robosim_wrapper.py`'s module docstring: the per-tick step is
raw float64 bytes both ways, everything else (reset, field params, get_state)
a JSON line. Replies are always `<kind byte><uint32 length><payload>` frames.
"""

import json
import os
import struct
import sys

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
# reads from only ever sees protocol frames.
_PROTOCOL_FD = os.dup(1)
_PROTOCOL_OUT = os.fdopen(_PROTOCOL_FD, "wb")
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)
os.close(_devnull_fd)

import numpy as np  # noqa: E402
import robosim  # noqa: E402


def _emit_frame(kind: bytes, body: bytes) -> None:
    _PROTOCOL_OUT.write(kind + struct.pack("<I", len(body)) + body)
    _PROTOCOL_OUT.flush()


def _emit(payload: dict) -> None:
    _emit_frame(b"J", json.dumps(payload).encode())


def _emit_state(state) -> None:
    # `get_state()` returns a list of Python floats (C doubles), so packing
    # them as little-endian float64 is exact -- the same values the JSON
    # path's shortest-repr round trip carried, without the formatting cost.
    _emit_frame(b"B", np.asarray(state, dtype="<f8").tobytes())


# Example: simple wrapper class
class SubprocessRSim:
    def __init__(self, sim_type, n_blue, n_yellow, field_type, time_step_ms):
        self._args = (sim_type, n_blue, n_yellow, field_type, time_step_ms)
        self.new_world()

    def new_world(self):
        """A brand-new native world, as at process start: nothing from the previous one survives."""
        sim_type, n_blue, n_yellow, field_type, time_step_ms = self._args
        self.sim = None  # destroy the old world before the new one is built
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

    stdin = sys.stdin.buffer
    try:
        while True:
            kind = stdin.read(1)
            if not kind:
                break
            try:
                if kind == b"S":
                    rows, cols = struct.unpack("<II", stdin.read(8))
                    body = stdin.read(rows * cols * 8)
                    commands = np.frombuffer(body, dtype="<f8").reshape(rows, cols)
                    sim.sim.step(commands)
                    _emit_state(sim.sim.get_state())
                    continue
                if kind != b"J":
                    # Framing is lost (nothing to resync on) -- say so and stop
                    # rather than guess where the next frame starts.
                    _emit({"error": f"unknown frame kind {kind!r}"})
                    break
                cmd = json.loads(stdin.readline())
                if "commands" in cmd:
                    # JSON form of the step command: the pre-binary protocol,
                    # kept as the reference the binary path is tested against.
                    state = sim.step(cmd["commands"])
                    _emit({"state": state})
                elif "reset" in cmd:
                    r = cmd["reset"]
                    sim.reset(r["ball_pos"], r["blue_robots_pos"], r["yellow_robots_pos"])
                    _emit({"ack": True})
                elif "new_world" in cmd:
                    sim.new_world()
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
