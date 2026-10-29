# ruff: noqa: E402
"""examples/real_attacker_defender.py — one attacker against one defender on a small field.

Run:
    pixi run python examples/real_attacker_defender.py            # real robots (MODE below)
    pixi run python examples/real_attacker_defender.py --mode rsim
    pixi run python examples/real_attacker_defender.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080: press Force Start to play, Halt to freeze
    pixi run python examples/real_attacker_defender.py --calibrate attacker   # or defender

What it does:
  - Two robots, one per team, both ours on one radio: the attacker attacks the goal in
    the camera's half, the defender defends it.
  - Everything to edit for a test day is in the "Test-day configuration" block: the
    field, which part of it the camera sees, each robot's IDs and working hardware, what
    each robot plays, and whether the referee's rules are on.
  - Nothing advances on its own: play only starts and stops from the dashboard. Every
    mode starts in HALT; Ctrl+C sends stop commands to the robots before exiting.
  - To change what a robot plays, or restart it, Halt and use the Live tab's Strategy panel
    (Run on the one running restarts it); no need to restart the script.

On the dashboard's Live tab, the field check must be all green or amber before Force Start:
a red row (robot not seen, outside the camera half, radio lost, wrong field size) halts the
robots at once. See docs/custom_referee_gui.md, "Field check".

Measuring a robot (`--calibrate attacker|defender`): that robot plays fixed moves instead
(`tactics/scripted_moves.py`, about 70 s: 10 s still, steps forward and back at 0.3, 0.6 and
1 m/s, sideways, turns) while the other stands still. Put it 1 m inside the area, facing
along the field's long side, and a ball at rest in view; Force Start. Halting (to put it
back) restarts the interrupted move. Then measure it from the recorded log:
    pixi run python tools/robot_calibration.py replays/field/<run>.log
Calibrate near the middle of the camera's view: its noise grows towards the image edges.
Run it twice per robot (early and late): how far the two runs disagree is how far to trust them.
Gain and loop delay should agree closely; acceleration, turning above all, is only rough.

The rest of a test day's measurements, any run (every real run records a log):
  - Floor friction: roll the ball by hand across open floor 8-10 times at different speeds,
    away from robots. Then convert the log and fit its free-rolling segments:
        pixi run python tools/convert_ssl_log.py replays/field/<run>.log --out replays/field/converted
        pixi run python -m utama_core.analysis.physics_ranges --real replays/field/converted
    The result replaces BALL_ROLL_DECEL (`config/settings.py`, the league's 0.29 m/s^2).
  - Kick speed: 5+ kicks per robot with the ball placed at its kicker (the attacker kicks in
    play); `robot_calibration` reports each kick's ball speed.
  - Occlusions, for the ball tracker: roll the ball behind a robot, let a robot dribble with it,
    cover it by hand. Note the times; they become test cases.

Before running on real robots:
  - ssl-vision's field geometry (length and width) must equal FIELD below, or the
    runner stops at the first geometry packet with "Field geometry mismatch".
  - The radio dongle is `PORT` in `utama_core/config/settings.py` (/dev/ttyACM0).
  - Real robots are capped at REAL_PARAMS.MAX_VEL (1 m/s, `config/robot_params.py`).
"""

import sys
from pathlib import Path

# Run as `pixi run python examples/<this>.py`: utama_core is not installed, so put the repo on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import dataclasses
import time
from typing import Callable, NamedTuple, Optional

from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import (
    AutoAdvanceConfig,
    GameConfig,
    RefereeProfile,
    RulesConfig,
    load_profile,
)
from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.dashboard.views import strategy as strategy_view
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.scripted_moves import ScriptedMovesTactic
from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

# ---------------------------------------------------------------------------
# Test-day configuration
# ---------------------------------------------------------------------------

MODE = "real"  # "real", "rsim" or "grsim"; --mode overrides it

# The lab field: 3 m x 2.25 m, goal 0.8 m wide, defense area 1 m wide x 0.5 m deep. Every
# value is a half-size in metres (half length, half width, ...), as in `STANDARD_FIELD_DIMS`;
# ssl-vision's field configuration must match it (lengths there are full sizes, in mm).
FIELD = FieldDimensions(
    full_field_half_length=1.5,
    full_field_half_width=1.125,
    half_defense_area_depth=0.25,
    half_defense_area_width=0.5,
    half_goal_width=0.4,
    center_circle_radius=0.3,
)

# The part of the field the camera sees: "left" or "right" half (x < 0 or x > 0 in
# ssl-vision's coordinates), or "full". The goal in that half is the one in play.
CAMERA_HALF = "right"

ATTACKER_COLOUR = "yellow"  # the defender is the other colour


class TestRobot(NamedTuple):
    vision_id: int  # its pattern on top
    firmware_id: int  # its radio ID; the two robots' must differ, one radio sends both
    # What works on it. A robot without a kicker pushes the ball instead of kicking; one
    # without a dribbler carries the ball slowly and never backs up; one without a ball
    # sensor (ir) has the ball once its centre is within 13 cm of it, whatever way it faces,
    # so it can stop ~2 cm short and kick at nothing (seen in grSim).
    profile: RobotProfile = RobotProfile(kicker=True, dribbler=True, ir=True)


# Swapping a robot is a change here only.
ATTACKER = TestRobot(vision_id=0, firmware_id=0)
DEFENDER = TestRobot(vision_id=1, firmware_id=1)

ATTACKER_TACTIC = "lead"  # see ATTACKER_TACTICS
DEFENDER_TACTIC = "goalkeeper"  # see DEFENDER_TACTICS

# "off": only goals are detected. "on": the `human` referee profile's rules (fouls, out of
# bounds, defense area ...) stop play and set up restarts; the operator still starts each
# restart, nothing advances on its own.
RULES = "off"

# With a camera (real, grsim), record every run as an SSL game log in replays/field/: raw
# vision, referee and radio bytes, for measuring the robots later (`GameLogWriter`).
RECORD = True

DASHBOARD_PORT = 8080

# ---------------------------------------------------------------------------

ATTACKER_TACTICS = {
    "lead": LeadAndSupportTactic,  # alone: takes the ball, turns to the open corner and shoots
    "give_and_go": GiveAndGoTactic,  # built to pass first; alone it plays like "lead"
}
DEFENDER_TACTICS = {
    "goalkeeper": None,  # the pinned GoalkeeperTactic: stays on the goal line
    "defense": DefenseTactic,  # shadows the shot from in front of the defense area
    "press": PressAndContainTactic,  # goes at the ball carrier
}

_NO_ROBOT = 15  # a robot ID no team uses here, for the goalkeeper slot nobody fills


def _field_bounds(field: FieldDimensions, half: str) -> FieldBounds:
    L, W = field.full_field_half_length, field.full_field_half_width
    if half == "right":
        return FieldBounds(top_left=(0.0, W), bottom_right=(L, -W))
    if half == "left":
        return FieldBounds(top_left=(-L, W), bottom_right=(0.0, -W))
    if half == "full":
        return field.full_field_bounds
    raise ValueError(f"CAMERA_HALF must be 'left', 'right' or 'full', got {half!r}")


def _play_or_hold(game, free_robots, prev_partition, available_tactic_ids) -> dict:
    """The robot plays its tactic; while that does not apply (press: no opponent near the ball),
    it holds the shot line instead of the kernel rejecting the pick."""
    if not free_robots:
        return {}
    return {("play" if "play" in available_tactic_ids else "hold"): free_robots}


def _one_robot_strategy(robot_id: int, tactic_cls) -> AbstractStrategy:
    """`robot_id` plays `tactic_cls`, or keeps goal when `tactic_cls` is None."""
    outfield = () if tactic_cls is None else (robot_id,)

    def _build(motion_controller) -> KernelSchedulerStrategy:
        # A Strategy needs one registered tactic even with no outfield robot to run it.
        return KernelSchedulerStrategy(
            tactics={"play": (tactic_cls or DefenseTactic)(), "hold": DefenseTactic()},
            partitioner=_play_or_hold,
            outfield_robot_ids=outfield,
            ctx=TickContext(motion_controller=motion_controller),
        )

    return AbstractStrategy(
        build_kernel_strategy=_build,
        goalkeeper_id=robot_id if tactic_cls is None else _NO_ROBOT,
    )


def _referee_profile() -> RefereeProfile:
    """`RULES`' profile; every transition is left to the operator either way."""
    if RULES == "on":
        human = load_profile("human")
        game = dataclasses.replace(human.game, half_duration_seconds=3600.0, kickoff_team=ATTACKER_COLOUR)
        return dataclasses.replace(human, profile_name="attacker_defender_test (rules on)", game=game)
    if RULES != "off":
        raise ValueError(f"RULES must be 'on' or 'off', got {RULES!r}")
    defaults = RulesConfig()
    rules = RulesConfig(
        **{
            f.name: dataclasses.replace(getattr(defaults, f.name), enabled=(f.name == "goal_detection"))
            for f in dataclasses.fields(RulesConfig)
        }
    )
    return RefereeProfile(
        profile_name="attacker_defender_test",
        rules=rules,
        game=GameConfig(
            half_duration_seconds=3600.0,
            kickoff_team=ATTACKER_COLOUR,
            force_start_after_goal=False,
            auto_advance=AutoAdvanceConfig(
                stop_to_next_command=False,
                prepare_kickoff_to_normal=False,
                prepare_penalty_to_normal=False,
                direct_free_to_normal=False,
                ball_placement_to_next=False,
                normal_start_to_force=False,
            ),
        ),
    )


def build_runner(
    mode: str, calibrate: Optional[str] = None, *, attacker_tactics: Optional[dict[str, Callable]] = None
) -> StrategyRunner:
    """Use the test-day configuration, optionally with another attacker's tactic factories."""
    attacker_name = ATTACKER_TACTIC if attacker_tactics is None else next(iter(attacker_tactics))
    attacker_tactics = ATTACKER_TACTICS if attacker_tactics is None else attacker_tactics
    real = mode == "real"
    # In the simulators each team's robots are numbered from 0.
    attacker_id = ATTACKER.vision_id if real else 0
    defender_id = DEFENDER.vision_id if real else 0
    attacker_is_yellow = ATTACKER_COLOUR == "yellow"
    if ATTACKER_COLOUR not in ("yellow", "blue"):
        raise ValueError(f"ATTACKER_COLOUR must be 'yellow' or 'blue', got {ATTACKER_COLOUR!r}")

    # The defender's goal is the one in the camera's half, so the attacker's own goal
    # is on the other side. With the full field, the attacker attacks the right goal.
    attacker_is_right = CAMERA_HALF == "left"

    attacker_map = {ATTACKER.vision_id: ATTACKER.firmware_id}
    defender_map = {DEFENDER.vision_id: DEFENDER.firmware_id}
    # The profiles apply in the simulators too, to rehearse a test with a broken robot.
    attacker_profiles = {attacker_id: ATTACKER.profile}
    defender_profiles = {defender_id: DEFENDER.profile}

    profile = _referee_profile()
    referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
    server = attach_dashboard(port=DASHBOARD_PORT)
    referee_view.attach(server, referee, profile)

    attacker_tactic, defender_tactic = attacker_tactics[attacker_name], DEFENDER_TACTICS[DEFENDER_TACTIC]
    if calibrate is not None:
        moves, still = ScriptedMovesTactic, lambda: ScriptedMovesTactic(script=())
        attacker_tactic, defender_tactic = (moves, still) if calibrate == "attacker" else (still, moves)
    game_log = (
        f"replays/field/{time.strftime('%Y%m%d_%H%M%S')}_{mode}{'_calibrate_' + calibrate if calibrate else ''}.log"
        if RECORD and mode != "rsim"
        else None
    )
    if game_log:
        print(f"Recording to {game_log}")
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=_one_robot_strategy(attacker_id, attacker_tactic),
        opp_strategy=_one_robot_strategy(defender_id, defender_tactic),
        my_team_is_yellow=attacker_is_yellow,
        my_team_is_right=attacker_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=1,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
        show_live_status=not real,
        yellow_vision_to_cmd_mapping=(attacker_map if attacker_is_yellow else defender_map) if real else None,
        blue_vision_to_cmd_mapping=(defender_map if attacker_is_yellow else attacker_map) if real else None,
        yellow_robot_profiles=attacker_profiles if attacker_is_yellow else defender_profiles,
        blue_robot_profiles=defender_profiles if attacker_is_yellow else attacker_profiles,
    )
    if calibrate is None:
        # The Live tab's Strategy panel: change either robot's tactic, or restart it, while halted.
        strategy_view.attach(
            server,
            runner,
            choices={
                "my": {n: (lambda t=t: _one_robot_strategy(attacker_id, t)) for n, t in attacker_tactics.items()},
                "opp": {n: (lambda t=t: _one_robot_strategy(defender_id, t)) for n, t in DEFENDER_TACTICS.items()},
            },
            running={"my": attacker_name, "opp": DEFENDER_TACTIC},
        )
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default=MODE)
    parser.add_argument("--calibrate", choices=("attacker", "defender"), help="measure this robot: fixed moves")
    args = parser.parse_args()
    if args.calibrate and not RECORD:
        parser.error("--calibrate needs RECORD = True: the measurement is read from the log")
    build_runner(args.mode, args.calibrate).run()


if __name__ == "__main__":
    main()
