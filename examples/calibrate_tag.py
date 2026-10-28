# ruff: noqa: E402
"""examples/calibrate_tag.py — where one real robot's vision tag sits on it: no dashboard.

Run:
    pixi run calibrate-tag --robot yellow:1 --radio 0     # colour:vision id (its pattern), radio id
    pixi run calibrate-tag --robot yellow:1 --radio 0 --mode rsim --yes   # rehearse it in the simulator

Put the robot near the middle of the camera's view, 0.5 m clear all round and 1 m clear ahead and
behind. It turns on the spot both ways at two rates, then drives straight forward and back twice
(about 70 s). Last it asks you to press its front flat against a straight edge (a ruler) laid
along a field line that runs across the field (along y), the robot facing +x, and holds still.

It prints, from the recorded log (`analysis/tag_calibration.py`):
- **Tag offset**: where the tag's centre sits from the robot's centre, from the circle it traces
  while the robot turns on the spot about its own centre.
- **Drive direction**: how far its straight drives run off the tag's heading: the tag's heading
  offset plus any sideways pull of the wheels.
And, measured here while the robot rests against the edge:
- **Heading offset**: the tag's heading minus the robot's true one (0, facing +x), as good as
  the edge is square to the field (1 mm in 5 cm is about 1 degree).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math

from calibrate_robot import build_runner, parse_args, play

from utama_core.analysis.robot_calibration import FieldLog
from utama_core.analysis.tag_calibration import report, tag_calibration
from utama_core.tactics.scripted_moves import Move

_REST = 1.5
_ALIGN = (
    "Press the robot's front flat against a straight edge laid along a field line that runs "
    "across the field (along y), the robot facing +x; leave it there"
)


def _spin(rate: float, turns: float) -> list[Move]:
    seconds = turns * 2 * math.pi / abs(rate)
    return [Move(f"turn {rate:+g} rad/s", seconds, turn=rate), Move("rest", _REST)]


def _drive(speed: float, seconds: float) -> list[Move]:
    return [Move(f"drive {speed:+g} m/s", seconds, forward=speed), Move("rest", _REST)]


TAG_SCRIPT: tuple[Move, ...] = (
    Move(
        "start",
        0.0,
        prompt="Robot near the middle of the camera's view facing +x, 0.5 m clear all round, 1 m ahead and behind",
    ),
    Move("still", 2.0),
    *_spin(1.0, 2.0),
    *_spin(-1.0, 2.0),
    *_spin(2.0, 2.0),
    *_spin(-2.0, 2.0),
    *_drive(0.3, 2.5),
    *_drive(-0.3, 2.5),
    *_drive(0.3, 2.5),
    *_drive(-0.3, 2.5),
    Move("against the edge", 0.0, prompt=_ALIGN),
    Move("aligned", 3.0),
)
_ALIGNED = len(TAG_SCRIPT) - 1


def main() -> None:
    args = parse_args(__doc__)
    runner, tactic, robot_id, label, game_log = build_runner(args, TAG_SCRIPT, "calibrate_tag")
    print(
        f"Calibrating {label}'s tag; " + (f"recording to {game_log}" if game_log else "a rehearsal, nothing recorded")
    )
    headings: list[float] = []

    def on_tick(step: int) -> None:
        robot = runner.my.game.friendly_robots.get(robot_id) if step == _ALIGNED else None
        if robot is not None:
            headings.append(robot.orientation)

    if not play(runner, tactic, TAG_SCRIPT, robot_id, label, args.yes, on_tick):
        return
    print(f"\a\nDONE: {label} has played the whole tag calibration.\n")
    heading_offset = None
    if headings:
        heading_offset = math.degrees(
            math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))
        )
        print(f"heading offset    {heading_offset:+.1f}° (the tag's heading against the edge, anticlockwise +)")
    if game_log:
        results = tag_calibration(FieldLog(game_log))
        print(report(results))
        out = Path(game_log).with_suffix(".tag.json")
        out.write_text(json.dumps({"log": game_log, "heading_offset_deg": heading_offset, "robots": results}, indent=1))
        print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
