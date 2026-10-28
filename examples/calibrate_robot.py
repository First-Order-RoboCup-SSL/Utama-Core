# ruff: noqa: E402
"""examples/calibrate_robot.py — measure one real robot from the terminal: no dashboard.

Run:
    pixi run calibrate --robot yellow:1 --radio 0     # colour:vision id (its pattern), radio id
    pixi run calibrate --robot yellow:1 --radio 0 --mode rsim --yes   # rehearse it in the simulator

It waits until vision sees the robot, then asks before each part that needs you, in this
terminal: put the robot near the middle of the camera's view facing +x (0.8 m clear ahead, 0.6 m
to its left) with a ball at rest in view; later put the ball against its dribbler, once to dribble
and once per kick (1.5 m clear ahead). Between those it plays fixed moves
(`tactics/scripted_moves.py`, `FULL_SCRIPT`): about 65 s of driving and turning, one kick with
no ball (a kicker left unused runs flat; this charges it), then holding, turning and driving with
the ball, then 3 kicks. Each move is printed as it starts. Ctrl+C stops the robot and exits at
any time.

When the script ends it says DONE, analyses the recorded log (`analysis/robot_calibration.py`)
and prints the robot's numbers: loop delay, speed gain and acceleration per axis, vision noise,
kick speed and direction off its heading, and how it holds the ball. They are written beside the
log as `<log>.calibration.json`. Calibrate every robot, twice (early and late in a session): how
far two runs disagree is how far to trust them.

The field and camera view are `FIELD` and `CAMERA_HALF` in `shooting_drill.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import threading
import time
from typing import Optional

from shooting_drill import (
    CAMERA_HALF,
    FIELD,
    TestRobot,
    _field_bounds,
    _one_robot_strategy,
)

from utama_core.analysis.robot_calibration import (
    FieldLog,
    ball_noise,
    calibrate,
    report,
)
from utama_core.run import StrategyRunner
from utama_core.tactics.scripted_moves import FULL_SCRIPT, ScriptedMovesTactic


def _robot(text: str) -> tuple[str, int]:
    colour, _, vision_id = text.partition(":")
    if colour not in ("yellow", "blue") or not vision_id.isdigit():
        raise argparse.ArgumentTypeError("expected yellow:<id> or blue:<id>, e.g. yellow:1")
    return colour, int(vision_id)


class _Prompt:
    """Asks one question in the terminal on a thread, so the runner keeps the robot still meanwhile."""

    def __init__(self):
        self.answered = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def asking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ask(self, text: str) -> None:
        self.answered.clear()

        def _wait():
            input(f"\n>>> {text}.\n    Press Enter when ready (Ctrl+C to stop) ")
            self.answered.set()

        self._thread = threading.Thread(target=_wait, daemon=True)
        self._thread.start()


def _wait_until_seen(runner: StrategyRunner, robot_id: int, label: str) -> None:
    print(f"Waiting for vision to see {label} ...")
    while True:
        runner.step_once()
        game = runner.my.game
        robot = game.friendly_robots.get(robot_id) if game is not None else None
        if robot is not None:
            ball = "ball at ({:.2f}, {:.2f})".format(game.ball.p.x, game.ball.p.y) if game.ball else "no ball seen"
            print(
                f"Seen {label} at ({robot.p.x:.2f}, {robot.p.y:.2f}) facing {math.degrees(robot.orientation):.0f}°; {ball}"
            )
            return


def _analyse(log_path: str) -> None:
    log = FieldLog(log_path)
    results, ball = calibrate(log), ball_noise(log.balls)
    print(report(results, ball))
    out = Path(log_path).with_suffix(".calibration.json")
    out.write_text(json.dumps({"log": log_path, "run_info": log.info, "robots": results, "ball": ball}, indent=1))
    print(f"\nWritten {out}")


def build_runner(args, script, name: str):
    """(runner, tactic, robot id, label, game log) playing `script` on the robot `args` names, recording
    to `replays/field/<stamp>_real_<name>_<colour><id>.log` on real robots."""
    colour, vision_id = args.robot
    real = args.mode == "real"
    robot = TestRobot(vision_id=vision_id, firmware_id=args.radio)
    label = f"{colour} {vision_id} (radio {args.radio})"
    is_yellow = colour == "yellow"
    robot_id = vision_id if real else 0  # the simulator numbers robots from 0
    mapping = {robot.vision_id: robot.firmware_id} if real else None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    game_log = f"replays/field/{stamp}_real_{name}_{colour}{vision_id}.log" if real else None
    tactic = ScriptedMovesTactic(script)
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=_one_robot_strategy(robot_id, lambda: tactic),
        my_team_is_yellow=is_yellow,
        my_team_is_right=False,
        mode=args.mode,
        exp_friendly=1,
        exp_enemy=0,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        show_live_status=False,
        yellow_vision_to_cmd_mapping=mapping if is_yellow else None,
        blue_vision_to_cmd_mapping=None if is_yellow else mapping,
        yellow_robot_profiles={robot_id: robot.profile} if is_yellow else None,
        blue_robot_profiles=None if is_yellow else {robot_id: robot.profile},
    )
    return runner, tactic, robot_id, label, game_log


def play(runner, tactic, script, robot_id: int, label: str, yes: bool, on_tick=None) -> bool:
    """Plays `script` to its end, asking in the terminal at each prompt (`yes`: answers them all).
    `on_tick(step)` after every tick. False if stopped with Ctrl+C; the robot is halted either way."""
    prompt = _Prompt()
    total = len(script)
    try:
        _wait_until_seen(runner, robot_id, label)
        shown = -1
        while not tactic.done:
            runner.step_once()
            if on_tick is not None and not tactic.done:
                on_tick(tactic.step)
            if tactic.waiting_for is not None:
                if yes:
                    print(f">>> {tactic.waiting_for}: yes")
                    tactic.proceed()
                elif prompt.answered.is_set():
                    prompt.answered.clear()
                    tactic.proceed()
                elif not prompt.asking:
                    prompt.ask(f"{script[tactic.step].name}: {tactic.waiting_for}")
            elif tactic.step != shown and not tactic.done:
                m = script[tactic.step]
                print(f"[{tactic.step + 1:>2}/{total}] {m.name} ({m.seconds:g} s)")
            shown = tactic.step
        return True
    except KeyboardInterrupt:
        print("\nStopped: the robot is halted. Nothing analysed.")
        return False
    finally:
        runner.close()


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=_robot, required=True, help="colour:vision id, e.g. yellow:1")
    parser.add_argument("--radio", type=int, required=True, help="the robot's radio (firmware) id")
    parser.add_argument("--mode", choices=("real", "rsim"), default="real", help="rsim: rehearse, nothing recorded")
    parser.add_argument("--yes", action="store_true", help="answer every question at once (a rehearsal)")
    return parser.parse_args()


def main() -> None:
    args = parse_args(__doc__)
    runner, tactic, robot_id, label, game_log = build_runner(args, FULL_SCRIPT, "calibrate")
    print(f"Calibrating {label}; " + (f"recording to {game_log}" if game_log else "a rehearsal, nothing recorded"))
    if play(runner, tactic, FULL_SCRIPT, robot_id, label, args.yes):
        print(f"\a\nDONE: {label} has played the whole calibration.")
        if game_log:
            print(f"Analysing {game_log} ...\n")
            _analyse(game_log)


if __name__ == "__main__":
    main()
