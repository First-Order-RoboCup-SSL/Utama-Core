"""Which of our robots are on the field, and the pattern on each one's lid: no pairing by hand.

A robot is its radio ID: the radio board stays with its motors, kicker and dribbler, and so does
its profile (`config/robots/radio_<id>.json`). The lid with the vision pattern is a separate
shell that can be swapped. So for each radio ID in turn it sends a still command and listens for
the robot's answer (its feedback frame, ~15 ms after a command addressed to it). Each radio that
answers is turned on the spot by about `TURN_RAD` and back, and the pattern whose heading turned
that much is that robot's lid. Only the robot being probed moves; it ends near where it started.

The result is saved to `FOUND_PATH` (one file per machine, not checked in), which the examples
read on real robots (`shooting_drill.on_field`) to know each robot's colour and vision ID. Run it
after switching robots on or swapping a lid:

    pixi run find-robots                 # radios 0 to 7
    pixi run find-robots --radios 0 1 4

It also says when a robot's lid is not the one its profile was calibrated with: the tag offset in
the profile is that lid's, so calibrate it again (`pixi run calibrate`).
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from pathlib import Path
from typing import Callable, Optional

from utama_core.config.settings import MULTICAST_GROUP, VISION_PORT
from utama_core.entities.data.command import RobotCommand
from utama_core.team_controller.src.controllers.common.robot_profile import (
    PROFILE_DIR,
    profile_path,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)

FOUND_PATH = Path(__file__).resolve().parents[5] / "replays" / "field" / "robots_found.json"
RADIOS = tuple(range(8))
TICK_S = 1 / 60
ANSWER_S = 0.3  # still, listening for the radio's answer and the lids' headings before the turn
TURN_SPEED = 2.0  # rad/s
TURN_S = 0.6  # about 1.2 rad with the robot's acceleration and delay
SETTLE_S = 0.4  # still after a turn, for the heading to settle and vision to catch up
TURN_RAD = 0.5  # a lid whose heading turned at least this much is the probed robot's
LOW_REPLY_RATE = 0.8  # a robot answering fewer of its commands than this has a poor radio link
LOOK_S = 0.5  # how long `confirm` watches vision for the saved lids
_STILL = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)

Lid = tuple[str, int]  # (colour, vision id)


def mean_heading(headings: list[float]) -> Optional[float]:
    if not headings:
        return None
    return math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))


def which_turned(before: dict[Lid, list[float]], after: dict[Lid, list[float]]) -> tuple[list[Lid], dict[Lid, float]]:
    """The lids whose mean heading turned `TURN_RAD` or more between `before` and `after`, and how
    far every lid seen both times turned (rad, anticlockwise +)."""
    turned = {}
    for lid in before.keys() & after.keys():
        b, a = mean_heading(before[lid]), mean_heading(after[lid])
        if b is not None and a is not None:
            turned[lid] = math.remainder(a - b, math.tau)
    return sorted(lid for lid, d in turned.items() if abs(d) >= TURN_RAD), turned


def vision_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((MULTICAST_GROUP, VISION_PORT))
    group = struct.pack("4sL", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group)
    sock.setblocking(False)
    return sock


def read_lids(sock: socket.socket, into: dict[Lid, list[float]]) -> None:
    """Every robot detection queued on `sock`: its heading, appended under its lid."""
    packet = SSL_WrapperPacket()
    while True:
        try:
            data = sock.recv(65536)
        except BlockingIOError:
            return
        packet.ParseFromString(data)
        if packet.HasField("detection"):
            for colour, robots in (("yellow", packet.detection.robots_yellow), ("blue", packet.detection.robots_blue)):
                for r in robots:
                    into.setdefault((colour, r.robot_id), []).append(r.orientation)


class _Probe:
    """Sends one radio its commands every tick while gathering vision and radio answers."""

    def __init__(self, controller, sock, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.controller, self.sock, self.clock, self.sleep = controller, sock, clock, sleep
        self.seen: set[Lid] = set()  # every lid seen while probing
        self.current: Optional[int] = None  # the radio being probed
        self.sent: dict[int, int] = {}  # radio -> commands sent to it
        self.replies: dict[int, int] = {}  # radio -> replies from it

    def hold(self, radio: int, command: RobotCommand, seconds: float) -> tuple[dict[Lid, list[float]], set[int]]:
        lids: dict[Lid, list[float]] = {}
        answered: set[int] = set()
        read_lids(self.sock, {})  # what queued before this step is stale
        self.current = radio
        end = self.clock() + seconds
        while self.clock() < end:
            self.controller.add_robot_commands(command, radio)
            self.controller.send_robot_commands()
            replied = {r.id for r in self.controller.get_robots_responses()}
            answered |= replied
            self.sent[radio] = self.sent.get(radio, 0) + 1
            self.replies[radio] = self.replies.get(radio, 0) + (radio in replied)
            read_lids(self.sock, lids)
            self.sleep(TICK_S)
        self.seen |= lids.keys()
        return lids, answered

    def radio(self, radio: int) -> dict:
        """One radio: whether it answered and, if it did, the lid that turned with it."""
        before, answered = self.hold(radio, _STILL, ANSWER_S)
        row = {"radio": radio, "answered": radio in answered, "lid": None, "turned": []}
        if not row["answered"]:
            return row
        turn = _STILL._replace(angular_vel=TURN_SPEED)
        self.hold(radio, turn, TURN_S)
        after, _ = self.hold(radio, _STILL, SETTLE_S)
        self.hold(radio, turn._replace(angular_vel=-TURN_SPEED), TURN_S)  # back to about where it was
        self.hold(radio, _STILL, SETTLE_S)
        row["reply_rate"] = round(self.replies[radio] / self.sent[radio], 2)
        turned, _ = which_turned(before, after)
        row["turned"] = [f"{c} {i}" for c, i in turned]
        if len(turned) == 1:
            row["lid"] = list(turned[0])
        return row


def find(radios=RADIOS, controller=None, sock=None, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Probes each of `radios`; the lids seen, and per radio whether it answered and its lid."""
    if controller is None:
        from utama_core.team_controller.src.controllers.real.real_robot_controller import (
            RealRobotController,
        )

        controller = RealRobotController(is_team_yellow=True, n_friendly=1)
    sock = vision_socket() if sock is None else sock
    probe = _Probe(controller, sock, clock, sleep)
    rows = []
    try:
        for radio in radios:
            row = probe.radio(radio)
            rows.append(row)
            print(f"  radio {radio}: " + _describe(row), flush=True)
    finally:
        if probe.current is not None:  # stopped mid-turn (Ctrl+C): stop that robot
            for _ in range(5):
                controller.add_robot_commands(_STILL, probe.current)
                controller.send_robot_commands()
        if hasattr(controller, "serial_port"):
            controller.serial_port.close()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "robots": rows,
        "lids_seen": sorted(f"{c} {i}" for c, i in probe.seen),
    }


def _describe(row: dict) -> str:
    if not row["answered"]:
        return "no answer"
    if row["lid"] is not None:
        return f"{row['lid'][0]} {row['lid'][1]}"
    if row["turned"]:
        return f"answered; several lids turned ({', '.join(row['turned'])}): run it again"
    return "answered, but no lid in view turned: its lid is out of view or unreadable"


def on_field(path: Path = FOUND_PATH) -> tuple[Optional[str], dict[int, Lid]]:
    """(when, {radio: lid}) from the last `find`; (None, {}) if it never ran on this machine."""
    if not path.exists():
        return None, {}
    found = json.loads(path.read_text())
    return found["time"], {r["radio"]: tuple(r["lid"]) for r in found["robots"] if r["lid"] is not None}


def in_view(seconds: float = LOOK_S, sock=None, clock=time.monotonic, sleep=time.sleep) -> set[Lid]:
    """The lids vision sees in `seconds`; nothing moves."""
    own = sock is None
    sock = vision_socket() if own else sock
    lids: dict[Lid, list[float]] = {}
    try:
        end = clock() + seconds
        while clock() < end:
            read_lids(sock, lids)
            sleep(TICK_S)
    finally:
        if own:
            sock.close()
    return set(lids)


def confirm(radios, path: Path = FOUND_PATH, look=in_view, finder=find) -> dict[int, Lid]:
    """{radio: lid} for `radios`: the saved pairing when each of their lids is in view now, else a
    new find of those radios (saved). A lid swapped between two of `radios` still looks right, as
    both lids stay in view: run `find-robots` after such a swap."""
    when, lids = on_field(path)
    seen = look()
    missing = [r for r in radios if lids.get(r) not in seen]
    if when is not None and not missing:
        print(f"Robots as find-robots paired them ({when}), lids all in view: " + _pairs(radios, lids))
        return {r: lids[r] for r in radios}
    why = "never ran here" if when is None else f"lid of radio {', '.join(map(str, missing))} not in view"
    print(f"Finding the robots ({why}): each one turns a little on the spot and back ...")
    found = finder(tuple(radios))
    save(found, path)
    lids = {r["radio"]: tuple(r["lid"]) for r in found["robots"] if r["lid"] is not None}
    unpaired = [r for r in radios if r not in lids]
    if unpaired:
        raise SystemExit(f"{report(found)}\n\nRadio {', '.join(map(str, unpaired))} not found on the field.")
    print("Found: " + _pairs(radios, lids))
    return {r: lids[r] for r in radios}


def _pairs(radios, lids: dict[int, Lid]) -> str:
    return ", ".join(f"radio {r} {lids[r][0]} {lids[r][1]}" for r in radios)


def calibrated_lid(radio: int, directory: Path = PROFILE_DIR) -> Optional[str]:
    """The lid the robot's profile was calibrated on (its `calibrated.lid`), e.g. "yellow 0"."""
    path = profile_path(radio, directory)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("calibrated", {}).get("lid")


def report(found: dict, directory: Path = PROFILE_DIR) -> str:
    rows = found["robots"]
    lines = [f"Robots on the field ({found['time']}):", f"  {'radio':<6} {'lid':<10} profile"]
    paired = set()
    for row in rows:
        if row["lid"] is None:
            continue
        lid = f"{row['lid'][0]} {row['lid'][1]}"
        paired.add(lid)
        was = calibrated_lid(row["radio"], directory)
        if not profile_path(row["radio"], directory).exists():
            note = "none: pixi run calibrate"
        elif was is not None and was != lid:
            note = f"calibrated on lid {was}: its tag offset is that lid's, calibrate again"
        else:
            note = "ok"
        lines.append(f"  {row['radio']:<6} {lid:<10} {note}")
    low = [f"radio {r['radio']} {r['reply_rate']:.0%}" for r in rows if r.get("reply_rate", 1) < LOW_REPLY_RATE]
    if low:
        lines.append(f"Few replies ({', '.join(low)} of commands): radio range, interference or a flat battery.")
    if not paired:
        lines.append("  (none)")
    for row in rows:
        if row["answered"] and row["lid"] is None:
            lines.append(f"  radio {row['radio']}: {_describe(row)}")
    silent = [str(r["radio"]) for r in rows if not r["answered"]]
    if silent:
        lines.append(f"No answer from radio {', '.join(silent)}: off, out of range, or none of ours.")
    others = [lid for lid in found["lids_seen"] if lid not in paired]
    if others:
        lines.append(f"Lids in view that no radio turned: {', '.join(others)}.")
    return "\n".join(lines)


def save(found: dict, path: Path = FOUND_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=1) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radios", type=int, nargs="+", default=list(RADIOS), help="radio IDs to try (default 0-7)")
    args = parser.parse_args(argv)
    print("Each robot that answers turns about 70 degrees on the spot and back: keep 15 cm clear round them.")
    found = find(tuple(args.radios))
    print(f"\n{report(found)}\n\nSaved to {save(found)}")


if __name__ == "__main__":
    main()
