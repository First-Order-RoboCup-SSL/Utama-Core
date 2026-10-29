# ruff: noqa: E402
"""examples/calibrate_robot.py — measure one real robot from the terminal: no dashboard.

Run:
    pixi run calibrate --robot yellow:0 --radio 0     # colour:vision id (its pattern), radio id
    pixi run calibrate --robot yellow:0 --radio 0 --ball-only   # the ball part again, no driving
    pixi run calibrate --robot yellow:0 --radio 0 --mode rsim --yes   # rehearse it in the simulator

It waits until vision sees the robot, then asks you once to put it near the middle of the
camera's view facing +x (0.8 m clear ahead, 0.6 m to its left) with a ball at rest in view.
It plays fixed moves first (`tactics/scripted_moves.py`, `CALIBRATION_SCRIPT`): about 65 s of
driving and turning. Then the ball part (`ball_trials.py`), where it fetches the ball itself:
- a carry ladder: forward, backward, sideways and turning with the ball at rising speeds,
  `--repeats` times each, a direction ending at the first speed that loses the ball;
- `--kicks` kicks along the field, after one with no ball (a kicker left unused runs flat).
It asks only when the ball is out of view or near the edge, as after most kicks: roll it back
towards the middle and press Enter. Each step is printed as it starts. Ctrl+C stops the robot
and exits at any time.

When it ends it says DONE, prints the carry ladder (the fastest speed each way that kept the
ball: what `RobotParams.CARRY_SPEED`/`CARRY_TURN` may be) and analyses the recorded log
(`analysis/robot_calibration.py`): loop delay, speed gain and acceleration per axis, vision noise,
kick speed, the kick's bend per cm the ball sits off-centre, and how it holds the ball. They are
written beside the log as `<log>.calibration.json`. Calibrate every robot, twice (early and late
in a session): how far two runs disagree is how far to trust them.

The robot's profile (`ROBOTS` in `shooting_drill.py`, by radio ID) corrects vision by where its
tag sits; measure that first (`pixi run calibrate-tag`). The field and camera view are `FIELD`
and `CAMERA_HALF` there.
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

from ball_trials import BallTrialsTactic
from shooting_drill import (
    CAMERA_HALF,
    FIELD,
    ROBOTS,
    TestRobot,
    _field_bounds,
    _one_robot_strategy,
    no_rules_referee,
)

from utama_core.analysis.robot_calibration import (
    FieldLog,
    ball_noise,
    calibrate,
    report,
)
from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.scripted_moves import (
    CALIBRATION_SCRIPT,
    Move,
    ScriptedMovesTactic,
)
from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

_START = (
    "Robot near the middle of the camera's view facing +x, 0.8 m clear ahead and 0.6 m to its left, "
    "a ball at rest in view"
)


class _InTurn(BaseTactic[None]):
    """`tactics` one after the other, each played to its end."""

    tag = TacticTag.MIXED

    def __init__(self, *tactics):
        self.tactics = tactics

    @property
    def current(self):
        return next((t for t in self.tactics if not t.done), self.tactics[-1])

    def initial_mem(self) -> None:
        return None

    @property
    def done(self) -> bool:
        return all(t.done for t in self.tactics)

    @property
    def waiting_for(self) -> Optional[str]:
        return self.current.waiting_for

    def proceed(self) -> None:
        self.current.proceed()

    @property
    def status(self) -> str:
        return self.current.status

    def tick(self, game, ctx, robot_ids, mem):
        return self.current.tick(game, ctx, robot_ids, mem)


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


def profile_of(radio: int) -> RobotProfile:
    """The profile of our robot on `radio` (`ROBOTS` in `shooting_drill.py`); a blank one if none is."""
    return next((r.profile for r in ROBOTS if r.firmware_id == radio), RobotProfile())


def build_runner(args, tactic, name: str, profile: RobotProfile = RobotProfile()):
    """(runner, robot id, label, game log) playing `tactic` on the robot `args` names, its vision
    corrected by `profile`, recording to `replays/field/<stamp>_real_<name>_<colour><id>.log` on
    real robots."""
    colour, vision_id = args.robot
    real = args.mode == "real"
    robot = TestRobot(vision_id=vision_id, firmware_id=args.radio, profile=profile)
    label = f"{colour} {vision_id} (radio {args.radio})"
    is_yellow = colour == "yellow"
    robot_id = vision_id if real else 0  # the simulator numbers robots from 0
    mapping = {robot.vision_id: robot.firmware_id} if real else None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    game_log = f"replays/field/{stamp}_real_{name}_{colour}{vision_id}.log" if real else None
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
        referee=no_rules_referee(int(is_yellow), int(not is_yellow)),
        referee_initial_command=RefereeCommand.FORCE_START,
        show_live_status=False,
        yellow_vision_to_cmd_mapping=mapping if is_yellow else None,
        blue_vision_to_cmd_mapping=None if is_yellow else mapping,
        yellow_robot_profiles={robot_id: robot.profile} if is_yellow else None,
        blue_robot_profiles=None if is_yellow else {robot_id: robot.profile},
    )
    return runner, robot_id, label, game_log


def _ball_back(runner) -> None:
    """A rehearsal's stand-in for the operator rolling the ball back: put it near the middle."""
    if runner.sim_controller is not None:
        (x0, y0), (x1, y1) = runner.field_bounds.top_left, runner.field_bounds.bottom_right
        runner.sim_controller.teleport_ball((x0 + x1) / 2 + 0.3, (y0 + y1) / 2 + 0.2)


def play(runner, tactic, robot_id: int, label: str, yes: bool, on_tick=None) -> bool:
    """Plays `tactic` (a `ScriptedMovesTactic`, `BallTrialsTactic` or both `_InTurn`) to its end,
    asking in the terminal at each prompt (`yes`: answers them all, and in a simulator puts the
    ball back itself). `on_tick(tactic)` after every tick. False if stopped with Ctrl+C; the robot
    is halted either way."""
    prompt = _Prompt()
    try:
        _wait_until_seen(runner, robot_id, label)
        shown = None
        while not tactic.done:
            runner.step_once()
            if on_tick is not None and not tactic.done:
                on_tick(tactic)
            if tactic.waiting_for is not None:
                if yes:
                    print(f">>> {tactic.waiting_for}: yes")
                    _ball_back(runner)
                    runner.step_once()  # the ball where it was put, seen, before carrying on
                    tactic.proceed()
                elif prompt.answered.is_set():
                    prompt.answered.clear()
                    tactic.proceed()
                elif not prompt.asking:
                    prompt.ask(f"{tactic.status}: {tactic.waiting_for}")
            elif tactic.status != shown and not tactic.done:
                shown = tactic.status
                print(shown)
        return True
    except KeyboardInterrupt:
        print("\nStopped: the robot is halted. Nothing analysed.")
        return False
    finally:
        runner.close()


def parse_args(description: str, ball: bool = False):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=_robot, required=True, help="colour:vision id, e.g. yellow:0")
    parser.add_argument("--radio", type=int, required=True, help="the robot's radio (firmware) id")
    parser.add_argument("--mode", choices=("real", "rsim"), default="real", help="rsim: rehearse, nothing recorded")
    parser.add_argument("--yes", action="store_true", help="answer every question at once (a rehearsal)")
    if ball:
        parser.add_argument("--ball-only", action="store_true", help="skip the driving: the ball part alone")
        parser.add_argument("--repeats", type=int, default=2, help="carry trials per speed (default 2)")
        parser.add_argument("--kicks", type=int, default=10, help="kicks (default 10)")
    return parser.parse_args()


def carry_report(trials: BallTrialsTactic) -> str:
    lines = ["carry ladder (the fastest each way that kept the ball every time, and each trial)"]
    for name, kept in trials.ladder.summary().items():
        rows = [r for r in trials.ladder.results if r["direction"] == name]
        unit = "rad/s" if name.startswith("turn") else "m/s"
        tried = ", ".join(f"{r['speed']:g}{'' if r['held'] else ' LOST'}" for r in rows)
        lines.append(f"  {name:<19} {'-' if kept is None else f'{kept:g} {unit}':<10} ({tried})")
    return "\n".join(lines)


def _analyse(log_path: str, robot: tuple[str, int], profile: RobotProfile, trials: BallTrialsTactic) -> None:
    log = FieldLog(log_path)
    log.correct_tags({robot: (profile.tag_forward_m, profile.tag_left_m, math.radians(profile.tag_heading_deg))})
    results, ball = calibrate(log), ball_noise(log.balls)
    print(report(results, ball))
    out = Path(log_path).with_suffix(".calibration.json")
    out.write_text(
        json.dumps(
            {
                "log": log_path,
                "run_info": log.info,
                "robots": results,
                "ball": ball,
                "carry_ladder": {"kept": trials.ladder.summary(), "trials": trials.ladder.results},
            },
            indent=1,
        )
    )
    print(f"\nWritten {out}")


def main() -> None:
    args = parse_args(__doc__, ball=True)
    profile = profile_of(args.radio)
    trials = BallTrialsTactic(_field_bounds(FIELD, CAMERA_HALF), repeats=args.repeats, kicks=args.kicks)
    script = (Move("start", 0.0, prompt=_START),) + (() if args.ball_only else CALIBRATION_SCRIPT)
    tactic = _InTurn(ScriptedMovesTactic(script), trials)
    runner, robot_id, label, game_log = build_runner(args, tactic, "calibrate", profile)
    print(f"Calibrating {label}; " + (f"recording to {game_log}" if game_log else "a rehearsal, nothing recorded"))
    print(f"Its profile: {profile}")
    if play(runner, tactic, robot_id, label, args.yes):
        print(f"\a\nDONE: {label} has played the whole calibration.\n")
        print(carry_report(trials))
        if game_log:
            print(f"\nAnalysing {game_log} ...\n")
            _analyse(game_log, args.robot, profile, trials)


if __name__ == "__main__":
    main()
