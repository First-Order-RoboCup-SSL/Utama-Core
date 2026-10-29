# ruff: noqa: E402
"""The start of a test day in one command: is vision up, is the radio there, which robots are on
the field with which lids, does each one answer its commands, and is each one's profile for the
lid it wears?

    pixi run field-check                 # radios 0 to 7
    pixi run field-check --radios 0 1

1. Vision (`vision_check.py`, 2 s): cameras and their packets per second, the field size, the ball
   and the lids in view. Nothing heard: it stops there.
2. Robots (`find-robots`): each radio that answers turns about 70 degrees on the spot and back;
   the lid that turned is its; the share of its commands it answered; its profile. The pairing is
   saved for the examples.

Each line starts with "ok" or "!!". What it does not check: batteries (the robots do not report
them) and kicker charge (`kicker_charge.py` keeps it up during play).
"""

import argparse
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vision_check import _socket, listen

from utama_core.config.settings import MULTICAST_GROUP, PORT, VISION_PORT
from utama_core.team_controller.src.controllers.real import find_robots

VISION_S = 2.0
MIN_CAMERA_HZ = 30  # a camera sending fewer packets per second than this is struggling


def vision_lines(heard: dict, seconds: float) -> list[str]:
    """`listen`'s result as checklist lines."""
    if not heard["senders"]:
        return [f"!! vision: nothing on {MULTICAST_GROUP}:{VISION_PORT} (docs/real_robot_setup.md, 'Vision')"]
    lines = []
    for camera, n in sorted(heard["per_camera"].items()):
        hz = n / seconds
        lines.append(f"{'ok' if hz >= MIN_CAMERA_HZ else '!!'} camera {camera}: {hz:.0f} packets/s")
    field = heard["field"]
    lines.append(f"ok field {field[0]:.0f} x {field[1]:.0f} mm" if field else "!! no field geometry from ssl-vision")
    things = {thing for thing, _ in heard["positions"]}
    lines.append("ok ball in view" if "ball" in things else "!! no ball in view")
    lids = sorted(t for t in things if t != "ball")
    lines.append(f"ok lids in view: {', '.join(lids)}" if lids else "!! no robot lids in view")
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radios", type=int, nargs="+", default=list(find_robots.RADIOS))
    args = parser.parse_args(argv)

    sock = _socket()
    try:
        heard = listen(sock, VISION_S)
    finally:
        sock.close()
    lines = vision_lines(heard, VISION_S)
    print("\n".join(lines))
    if not heard["senders"]:
        sys.exit(1)

    print("\nFinding the robots: each one that answers turns about 70 degrees and back; keep 15 cm clear.")
    try:
        found = find_robots.find(tuple(args.radios))
    except (ConnectionError, socket.error) as e:
        print(f"!! radio: {e} (is the transmitter plugged in at {PORT}?)")
        sys.exit(1)
    find_robots.save(found)
    print("\n" + find_robots.report(found))


if __name__ == "__main__":
    main()
