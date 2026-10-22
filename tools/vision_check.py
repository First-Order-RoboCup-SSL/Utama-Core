# ruff: noqa: E402
"""Does ssl-vision's multicast reach this machine? Listens for a few seconds, with no radio or
runner, and prints who sent it, packets per second per camera, what was detected and the field
size ssl-vision announces. The first check on a test day (`docs/real_robot_setup.md`).

Each robot and ball is also listed where each camera puts it (mean over the run) and, when
several cameras see it, the largest gap between them: cameras whose calibrations agree are a few
cm apart. Rerun after each calibration change.

With `--guided ball` (or `--guided "blue 2"`) it checks the cameras against known spots instead:
it asks for the ball (or that robot, facing +x) on each of 9 points (corners, edge midpoints,
centre of the field ssl-vision announces), averages each camera there for `--seconds`, and prints
each camera's error at each point. Cameras that agree with each other can both be wrong; this
says by how much. The results are saved to `replays/field/` as the baseline for the next setup.

    pixi run python tools/vision_check.py [--seconds 5]
    pixi run python tools/vision_check.py --guided ball [--seconds 2]
"""

import argparse
import collections
import json
import math
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

GUIDE_RADIUS = 500  # mm: in the guided check, a detection further than this from the spot is not it


def _socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((MULTICAST_GROUP, VISION_PORT))
    group = struct.pack("4sL", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group)
    sock.settimeout(0.5)
    return sock


def listen(sock, seconds: float) -> dict:
    """What arrives in `seconds`: senders, packets and detections per camera, the field size and
    `positions`, (thing, camera) -> [(x, y, orientation)] in mm and rad (orientation None for a ball),
    and `ball_frames`, camera -> every ball of each of its frames ([[(x, y)]])."""
    per_camera, seen, senders, field = collections.Counter(), collections.Counter(), set(), None
    positions, ball_frames = collections.defaultdict(list), collections.defaultdict(list)
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
                    positions[(f"{colour} {r.robot_id}", d.camera_id)].append((r.x, r.y, r.orientation))
            ball_frames[d.camera_id].append([(b.x, b.y) for b in d.balls])
            if len(d.balls) == 1:  # with several balls in one image, which is which is unknown
                positions[("ball", d.camera_id)].append((d.balls[0].x, d.balls[0].y, None))
        if packet.HasField("geometry"):
            field = (packet.geometry.field.field_length, packet.geometry.field.field_width)
    return {
        "per_camera": per_camera,
        "seen": seen,
        "senders": senders,
        "field": field,
        "positions": positions,
        "ball_frames": ball_frames,
    }


def mean_pose(samples: list) -> tuple:
    """Mean (x, y, orientation) of `samples`; the orientation a circular mean, None for a ball."""
    x = sum(s[0] for s in samples) / len(samples)
    y = sum(s[1] for s in samples) / len(samples)
    if samples[0][2] is None:
        return x, y, None
    return x, y, math.atan2(sum(math.sin(s[2]) for s in samples), sum(math.cos(s[2]) for s in samples))


def guide_points(length: float, width: float) -> list:
    """The 9 spots (mm) the guided check asks for: corners, edge midpoints and centre."""
    return [(x * length / 2, y * width / 2) for y in (1, 0, -1) for x in (-1, 0, 1)]


def nearest_balls(frames: list, x: float, y: float, limit: float = GUIDE_RADIUS) -> list:
    """Of each frame's balls, the one nearest (`x`, `y`) when within `limit` mm, as
    (x, y, None): a stray detection elsewhere (something orange) is left out."""
    picked = []
    for balls in frames:
        near = [b for b in balls if math.hypot(b[0] - x, b[1] - y) <= limit]
        if near:
            bx, by = min(near, key=lambda b: math.hypot(b[0] - x, b[1] - y))
            picked.append((bx, by, None))
    return picked


def check(seconds: float) -> None:
    sock = _socket()
    heard = listen(sock, seconds)
    if not heard["senders"]:
        print(f"Nothing on {MULTICAST_GROUP}:{VISION_PORT} in {seconds:.0f} s: see docs/real_robot_setup.md, 'Vision'.")
        sys.exit(1)
    print(f"From: {', '.join(sorted(heard['senders']))}")
    per_camera = heard["per_camera"]
    for camera, n in sorted(per_camera.items()):
        print(f"Camera {camera}: {n / seconds:.0f} packets/s")
    print(
        "Detections per packet: "
        + ", ".join(f"{k} {v / max(1, sum(per_camera.values())):.1f}" for k, v in heard["seen"].items())
    )
    field = heard["field"]
    print(f"Field: {field[0]:.0f} x {field[1]:.0f} mm" if field else "Field: no geometry packet yet")

    by_thing = collections.defaultdict(dict)
    for (thing, camera), samples in heard["positions"].items():
        by_thing[thing][camera] = mean_pose(samples)
    for thing, cameras in sorted(by_thing.items()):
        where = ", ".join(f"camera {c} ({p[0]:.0f}, {p[1]:.0f})" for c, p in sorted(cameras.items()))
        gaps = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in combinations(cameras.values(), 2)]
        print(f"{thing}: {where}" + (f", {max(gaps):.0f} mm apart" if gaps else ""))


def guided(thing: str, seconds: float) -> None:
    sock = _socket()
    field = None
    while field is None:  # ssl-vision sends the geometry about once a second
        heard = listen(sock, 2.0)
        if not heard["senders"]:
            print(f"Nothing on {MULTICAST_GROUP}:{VISION_PORT}: run without --guided to diagnose.")
            sys.exit(1)
        field = heard["field"]
    print(f"Field {field[0]:.0f} x {field[1]:.0f} mm. Checking {thing!r} at 9 points, {seconds:.0f} s each.")
    if thing != "ball":
        print("Face the robot towards +x at each point.")

    results = []
    for i, (x, y) in enumerate(guide_points(*field), 1):
        input(f"\n[{i}/9] Put the {thing} at ({x:.0f}, {y:.0f}) mm, then press Enter ")
        heard = listen(sock, seconds)
        point = {"truth": [x, y], "cameras": {}}
        if thing == "ball":
            seen = {c: nearest_balls(f, x, y) for c, f in heard["ball_frames"].items()}
        else:
            seen = {c: s for (name, c), s in heard["positions"].items() if name == thing}
        for camera, samples in sorted(seen.items()):
            if not samples:
                continue
            mx, my, mo = mean_pose(samples)
            point["cameras"][camera] = {"seen": [mx, my], "orientation": mo, "frames": len(samples)}
            line = f"  camera {camera}: ({mx:.0f}, {my:.0f}), off by {math.hypot(mx - x, my - y):.0f} mm"
            line += f" (dx {mx - x:+.0f}, dy {my - y:+.0f})"
            if mo is not None:
                line += f", heading {math.degrees(mo):+.1f} deg"
            print(line)
        if not point["cameras"]:
            print(f"  no camera saw the {thing}")
        results.append(point)

    print("\nPer camera over the points it saw: mean error, worst, mean shift (dx, dy)")
    for camera in sorted({c for p in results for c in p["cameras"]}):
        errs = [
            (p["cameras"][camera]["seen"][0] - p["truth"][0], p["cameras"][camera]["seen"][1] - p["truth"][1])
            for p in results
            if camera in p["cameras"]
        ]
        dists = [math.hypot(dx, dy) for dx, dy in errs]
        print(
            f"  camera {camera}: {len(errs)}/9 points, {sum(dists) / len(dists):.0f} mm mean, {max(dists):.0f} worst,"
            f" shift ({sum(e[0] for e in errs) / len(errs):+.0f}, {sum(e[1] for e in errs) / len(errs):+.0f})"
        )
    out = (
        REPO_ROOT
        / "replays"
        / "field"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_vision_truth_{thing.replace(' ', '_')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"thing": thing, "field": field, "seconds": seconds, "points": results}, indent=1))
    print(f"Saved {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, help="how long to listen (default 5, or 2 per point with --guided)")
    parser.add_argument("--guided", metavar="THING", help='"ball" or a robot ("blue 2"): check it on 9 known points')
    args = parser.parse_args()
    if args.guided:
        guided(args.guided, args.seconds or 2.0)
    else:
        check(args.seconds or 5.0)


if __name__ == "__main__":
    main()
