# ruff: noqa: E402
"""Does ssl-vision's multicast reach this machine? Listens for a few seconds, with no radio or
runner, and prints who sent it, packets per second per camera, what was detected and the field
size ssl-vision announces. The first check on a test day (`docs/real_robot_setup.md`).

Each robot and ball is also listed where each camera puts it (mean over the run) and, when
several cameras see it, the largest gap between them: cameras whose calibrations agree are a few
cm apart. Rerun after each calibration change.

    pixi run python tools/vision_check.py [--seconds 5]
"""

import argparse
import collections
import socket
import struct
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.config.settings import MULTICAST_GROUP, VISION_PORT
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=5.0)
    seconds = parser.parse_args().seconds

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((MULTICAST_GROUP, VISION_PORT))
    group = struct.pack("4sL", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group)
    sock.settimeout(0.5)

    per_camera, seen, senders, field = collections.Counter(), collections.Counter(), set(), None
    positions = collections.defaultdict(list)  # (thing, camera) -> [(x, y)] in mm
    end = time.time() + seconds
    while time.time() < end:
        try:
            data, sender = sock.recvfrom(65536)
        except socket.timeout:
            continue
        senders.add(sender[0])
        packet = SSL_WrapperPacket()
        packet.ParseFromString(data)
        if packet.HasField("detection"):
            d = packet.detection
            per_camera[d.camera_id] += 1
            seen["yellow"] += len(d.robots_yellow)
            seen["blue"] += len(d.robots_blue)
            seen["balls"] += len(d.balls)
            for colour, robots in (("yellow", d.robots_yellow), ("blue", d.robots_blue)):
                for r in robots:
                    positions[(f"{colour} {r.robot_id}", d.camera_id)].append((r.x, r.y))
            if len(d.balls) == 1:  # with several balls in one image, which is which is unknown
                positions[("ball", d.camera_id)].append((d.balls[0].x, d.balls[0].y))
        if packet.HasField("geometry"):
            field = (packet.geometry.field.field_length, packet.geometry.field.field_width)

    if not senders:
        print(f"Nothing on {MULTICAST_GROUP}:{VISION_PORT} in {seconds:.0f} s: see docs/real_robot_setup.md, 'Vision'.")
        sys.exit(1)
    print(f"From: {', '.join(sorted(senders))}")
    for camera, n in sorted(per_camera.items()):
        print(f"Camera {camera}: {n / seconds:.0f} packets/s")
    print(
        "Detections per packet: "
        + ", ".join(f"{k} {v / max(1, sum(per_camera.values())):.1f}" for k, v in seen.items())
    )
    print(f"Field: {field[0]:.0f} x {field[1]:.0f} mm" if field else "Field: no geometry packet yet")

    by_thing = collections.defaultdict(dict)
    for (thing, camera), xys in positions.items():
        by_thing[thing][camera] = tuple(sum(c) / len(xys) for c in zip(*xys))
    for thing, cameras in sorted(by_thing.items()):
        where = ", ".join(f"camera {c} ({x:.0f}, {y:.0f})" for c, (x, y) in sorted(cameras.items()))
        gaps = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in combinations(cameras.values(), 2)]
        print(f"{thing}: {where}" + (f", {max(gaps):.0f} mm apart" if gaps else ""))


if __name__ == "__main__":
    main()
