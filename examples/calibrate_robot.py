# ruff: noqa: E402
"""examples/calibrate_robot.py — measure one real robot and write its profile: no dashboard.

Run:
    pixi run calibrate                  # finds the robots on the field; asks which if several
    pixi run calibrate --radio 0        # the robot on radio 0, whatever lid it wears
    pixi run calibrate --radio 0 --ball-only   # the ball part again, no driving
    pixi run calibrate --radio 0 --check       # is the new profile better?
    pixi run calibrate --mode rsim --yes       # rehearse it in the simulator

Without both `--robot colour:id` (its lid's pattern) and `--radio`, it looks the robot up
(`find_robots.py`): with `--radio`, the lid last found for it if that is in view; otherwise it finds
the robots on the field (each radio that answers turns a little on the spot and back) and takes
the robot wearing `--robot`, the only one, or the one you pick. It waits until vision sees the robot, then asks you once to put it near the middle of the
camera's view facing +x (0.8 m clear ahead, 0.6 m to its left) with a ball at rest in view.
It plays fixed moves first (`tactics/scripted_moves.py`, `CALIBRATION_SCRIPT`): about 65 s of
driving and turning, its turns on the spot also showing where its tag sits. Then the ball part
(`ball_trials.py`), where it fetches the ball itself:
- a carry ladder: forward, backward, sideways and turning with the ball at rising speeds,
  `--repeats` times each, a direction ending at the first speed that loses the ball;
- `--kicks` kicks along the field, each with the kicker charged.
It asks only when the ball is out of view or near the edge, as after most kicks: roll it back
towards the middle and press Enter. Each step is printed as it starts. Ctrl+C stops the robot
and exits at any time.

When it ends it says DONE, analyses the recorded log (`analysis/robot_calibration.py`: loop
delay, speed gain and acceleration per axis, vision noise, kicks, how it holds the ball; written
beside the log as `<log>.calibration.json`) and writes what it measured into the robot's profile
file, `utama_core/config/robots/radio_<radio>.json` (`RobotProfile`):
- `tag_forward_m`, `tag_left_m`: where its tag sits (`analysis/tag_calibration.py`);
- `kick_bend_deg_per_cm`, `kick_bias_deg`: its kicks' bend and bias, fitted together;
- `carry_speed`, `carry_turn`: the slowest direction's fastest speed that kept the ball.
What it could not measure stays as it was. Review the change with `git diff`.

`--check` then says whether the new profile does better than the committed one (`git show
HEAD:...`): `--blocks` blocks alternating the two (committed first), each carrying the ball every
way at that profile's caps `--repeats` times and kicking `--kicks` times at the middle of the far
end, aimed against its kick bend and bias. It prints per profile how often each carry kept the
ball and how far the kicks missed. Commit the file if the new one is no worse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import subprocess
import threading
import time
from dataclasses import replace
from typing import Optional

from ball_trials import BallTrialsTactic, at_caps
from shooting_drill import (
    CAMERA_HALF,
    FIELD,
    KEEP_OUT,
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
from utama_core.analysis.tag_calibration import tag_calibration
from utama_core.config.robot_params import REAL_PARAMS
from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.scripted_moves import (
    CALIBRATION_SCRIPT,
    Move,
    ScriptedMovesTactic,
)
from utama_core.team_controller.src.controllers.common.robot_profile import (
    RobotProfile,
    load_profile,
    profile_path,
    save_profile,
)
from utama_core.team_controller.src.controllers.real import find_robots

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
        keep_out=KEEP_OUT,
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
    parser.add_argument("--robot", type=_robot, help="its lid, colour:vision id, e.g. yellow:0 (default: found)")
    parser.add_argument("--radio", type=int, help="the robot's radio (firmware) id (default: found)")
    parser.add_argument("--mode", choices=("real", "rsim"), default="real", help="rsim: rehearse, nothing recorded")
    parser.add_argument("--yes", action="store_true", help="answer every question at once (a rehearsal)")
    if ball:
        parser.add_argument("--ball-only", action="store_true", help="skip the driving: the ball part alone")
        parser.add_argument("--check", action="store_true", help="compare the profile file with the committed one")
        parser.add_argument("--blocks", type=int, default=4, help="--check: blocks, alternating (default 4)")
        parser.add_argument("--repeats", type=int, default=2, help="carry trials per speed (default 2)")
        parser.add_argument("--kicks", type=int, help="kicks (default 10; --check: per block, default 5)")
    return parser.parse_args()


def carry_report(trials: BallTrialsTactic) -> str:
    lines = ["carry ladder (the fastest each way that kept the ball every time, and each trial)"]
    for name, kept in trials.ladder.summary().items():
        rows = [r for r in trials.ladder.results if r["direction"] == name]
        unit = "rad/s" if name.startswith("turn") else "m/s"
        tried = ", ".join(f"{r['speed']:g}{'' if r['held'] else ' LOST'}" for r in rows)
        lines.append(f"  {name:<19} {'-' if kept is None else f'{kept:g} {unit}':<10} ({tried})")
    return "\n".join(lines)


CARRY_TRANSLATIONS = ("forward", "backward", "left", "right")
CARRY_TURNS = ("turn anticlockwise", "turn clockwise")


def carry_caps(kept: dict[str, Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    """(carry speed, carry turn) from a ladder's `summary()`: the slowest direction's fastest
    speed that kept the ball; None where a direction was not tried or lost it at its slowest."""

    def slowest(names):
        speeds = [kept.get(n) for n in names]
        return None if any(v is None for v in speeds) else min(speeds)

    return slowest(CARRY_TRANSLATIONS), slowest(CARRY_TURNS)


def measured_profile(
    old: RobotProfile, tag: Optional[dict], bend: Optional[dict], kept: dict
) -> tuple[RobotProfile, dict]:
    """`old` with what this run measured in place of its numbers, and a note per part saying where
    each came from (or that it was not measured and is kept)."""
    changes, notes = {}, {}
    if tag is not None and tag["tag_forward_m"] is not None:
        changes.update(tag_forward_m=round(tag["tag_forward_m"], 3), tag_left_m=round(tag["tag_left_m"], 3))
        notes["tag"] = f"{len(tag['spins'])} turns on the spot"
    if bend is not None:
        changes.update(
            kick_bend_deg_per_cm=round(bend["deg_per_cm"], 1),
            kick_bias_deg=round(bend["bias_deg"], 1),
            kick_speed=bend["speed"],
            kick_spread_deg=bend["rms_deg"],
        )
        notes["kick"] = (
            f"{bend['kicks']} kicks: {bend['deg_per_cm']} +/- {bend['deg_per_cm_se']} deg/cm, "
            f"{bend['bias_deg']:+} +/- {bend['bias_se_deg']} deg, rms {bend['rms_deg']}, {bend['speed']} m/s"
        )
    speed, turn = carry_caps(kept)
    if speed is not None:
        changes["carry_speed"] = speed
    if turn is not None:
        changes["carry_turn"] = turn
    if kept:
        notes["carry"] = ", ".join(f"{name} {'-' if v is None else f'{v:g}'}" for name, v in kept.items())
    return replace(old, **changes), notes


def profile_diff(old: RobotProfile, new: RobotProfile) -> str:
    rows = [f"  {f}: {getattr(old, f)} -> {getattr(new, f)}" for f in vars(old) if getattr(old, f) != getattr(new, f)]
    return "\n".join(rows) if rows else "  (unchanged)"


def _analyse(log_path: str, robot: tuple[str, int], radio: int, old: RobotProfile, trials: BallTrialsTactic) -> None:
    """Analyses the run's log and writes what it measured into the robot's profile file."""
    log = FieldLog(log_path)
    tag = next((r for r in tag_calibration(log) if r["firmware_id"] == radio), None)  # from the raw tag
    offset = (old.tag_forward_m, old.tag_left_m)
    if tag is not None and tag["tag_forward_m"] is not None:
        offset = (tag["tag_forward_m"], tag["tag_left_m"])
    log.correct_tags({robot: offset})
    results, ball = calibrate(log), ball_noise(log.balls)
    print(report(results, ball))
    kept = trials.ladder.summary()
    out = Path(log_path).with_suffix(".calibration.json")
    out.write_text(
        json.dumps(
            {
                "log": log_path,
                "run_info": log.info,
                "robots": results,
                "ball": ball,
                "tag": tag,
                "carry_ladder": {"kept": kept, "trials": trials.ladder.results},
            },
            indent=1,
        )
    )
    print(f"\nWritten {out}")

    bend = next((r.get("kick_bend") for r in results if r.get("firmware_id") == radio), None)
    new, notes = measured_profile(old, tag, bend, kept)
    path = profile_path(radio)
    calibrated = json.loads(path.read_text()).get("calibrated", {}) if path.exists() else {}
    calibrated.update({"lid": f"{robot[0]} {robot[1]}", "log": Path(log_path).name, **notes})
    save_profile(radio, new, calibrated)
    print(f"\nRadio {radio}'s profile, {path}:\n{profile_diff(old, new)}")
    print("Review it with `git diff`; `pixi run calibrate ... --check` tries it against the committed one.")


def committed_profile(radio: int) -> Optional[RobotProfile]:
    """The robot's profile as last committed (`git show HEAD:...`); None if it has none."""
    path = profile_path(radio)
    rel = path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    shown = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, text=True)
    return RobotProfile(**json.loads(shown.stdout)["profile"]) if shown.returncode == 0 else None


def check_summary(blocks: list[dict]) -> str:
    """Per profile: how often each carry kept the ball, and how far the kicks missed."""
    lines = []
    for which in ("committed", "new"):
        rows = [b for b in blocks if b["profile"] == which]
        if not rows:
            continue
        p = rows[0]["caps"]
        lines.append(f"{which}: carry caps {p[0]:g} m/s, {p[1]:g} rad/s")
        carries = [t for b in rows for t in b["carry"]]
        for name in dict.fromkeys(t["direction"] for t in carries):
            held = [t["held"] for t in carries if t["direction"] == name]
            lines.append(f"  carry {name:<19} kept {sum(held)}/{len(held)}")
        misses = [k["miss_deg"] for b in rows for k in b["kicks"]]
        seen = [m for m in misses if m is not None]
        if seen:
            mean_abs = sum(abs(m) for m in seen) / len(seen)
            se = (sum((abs(m) - mean_abs) ** 2 for m in seen) / max(len(seen) - 1, 1) / len(seen)) ** 0.5
            lines.append(
                f"  kicks {len(misses)}: missed by {mean_abs:.1f} ± {se:.1f}° on average (mean of |miss|), "
                f"{sum(seen) / len(seen):+.1f}° to the left on average"
                + (f", {len(misses) - len(seen)} never left" if len(seen) < len(misses) else "")
            )
    return "\n".join(lines)


def run_check(args, bounds) -> None:
    """`--check`: blocks alternating the committed and the new profile; see the module."""
    new = load_profile(args.radio)
    old = committed_profile(args.radio)
    if old is None or old == new:
        print("Nothing to check: the profile file is " + ("not committed." if old is None else "as committed."))
        return
    print(f"Checking radio {args.radio}'s profile; changed from the committed one:\n{profile_diff(old, new)}")
    kicks = 5 if args.kicks is None else args.kicks
    blocks: list[dict] = []
    for block in range(args.blocks):
        which, profile = ("committed", old) if block % 2 == 0 else ("new", new)
        caps = (
            REAL_PARAMS.CARRY_SPEED if profile.carry_speed is None else profile.carry_speed,
            REAL_PARAMS.CARRY_TURN if profile.carry_turn is None else profile.carry_turn,
        )
        trials = BallTrialsTactic(bounds, repeats=args.repeats, kicks=kicks, directions=at_caps(*caps), aim=True)
        runner, robot_id, label, game_log = build_runner(args, trials, f"calibrate_check_{which}", profile)
        print(
            f"\nblock {block + 1}/{args.blocks}: the {which} profile"
            + (f", recording to {game_log}" if game_log else "")
        )
        if not play(runner, trials, robot_id, label, args.yes):
            break
        blocks.append({"profile": which, "caps": caps, "carry": trials.ladder.results, "kicks": trials.kick_results})
    if not blocks:
        return
    print("\n" + check_summary(blocks))
    out = Path(f"replays/field/{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}_calibrate_check_radio{args.radio}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"committed": vars(old), "new": vars(new), "blocks": blocks}, indent=1))
    print(f"\nWritten {out}")


def pick(lids: dict[int, tuple[str, int]], radio: Optional[int], robot: Optional[tuple[str, int]], ask=input) -> int:
    """The radio to calibrate among those found (`lids`): `radio`, the one wearing `robot`, the
    only one, or the one the operator picks."""
    if radio is None and robot is not None:
        radio = next((r for r, lid in lids.items() if lid == tuple(robot)), None)
        if radio is None:
            raise SystemExit(f"No robot found wearing {robot[0]} {robot[1]}.")
    if radio is None:
        if not lids:
            raise SystemExit("No robot found: switched on, in view, its radio in range?")
        if len(lids) == 1:
            return next(iter(lids))
        choices = ", ".join(f"{r} ({c} {v})" for r, (c, v) in sorted(lids.items()))
        while radio not in lids:
            answer = ask(f"Which radio to calibrate? {choices}: ").strip()
            radio = int(answer) if answer.isdigit() else None
    if radio not in lids:
        raise SystemExit(f"Radio {radio} was not found on the field.")
    return radio


def choose_robot(args) -> None:
    """Fills in `args.robot` and `args.radio`: on real robots from the robots found on the field
    unless both were given; in a rehearsal yellow 0 on radio 0."""
    if args.mode != "real":
        args.robot = args.robot or ("yellow", 0)
        args.radio = 0 if args.radio is None else args.radio
        return
    if args.robot is not None and args.radio is not None:
        return
    if args.radio is not None:
        args.robot = find_robots.confirm([args.radio])[args.radio]
        return
    print("Finding the robots on the field: each one that answers turns a little and back ...")
    found = find_robots.find()
    find_robots.save(found)
    print(find_robots.report(found) + "\n")
    lids = {r["radio"]: tuple(r["lid"]) for r in found["robots"] if r["lid"] is not None}
    args.radio = pick(lids, args.radio, args.robot)
    args.robot = lids[args.radio]


def main() -> None:
    args = parse_args(__doc__, ball=True)
    choose_robot(args)
    bounds = _field_bounds(FIELD, CAMERA_HALF)
    if args.check:
        run_check(args, bounds)
        return
    profile = load_profile(args.radio)
    trials = BallTrialsTactic(bounds, repeats=args.repeats, kicks=10 if args.kicks is None else args.kicks)
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
            _analyse(game_log, args.robot, args.radio, profile, trials)
        else:
            new, _ = measured_profile(profile, None, None, trials.ladder.summary())
            print(f"\nA rehearsal: nothing written. From the ladder alone it would be:\n{profile_diff(profile, new)}")


if __name__ == "__main__":
    main()
