"""examples/real_attacker_defender.py — one attacker against one defender on a small field.

Run:
    pixi run python examples/real_attacker_defender.py            # real robots (MODE below)
    pixi run python examples/real_attacker_defender.py --mode rsim
    pixi run python examples/real_attacker_defender.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080: press Force Start to play, Halt to freeze

What it does:
  - Two robots, one per team, both ours on one radio: the attacker attacks the goal in
    the camera's half, the defender defends it.
  - Everything to edit for a test day is in the "Test-day configuration" block: the
    field, which part of it the camera sees, each robot's vision and firmware IDs, and
    what each robot plays.
  - The referee runs with every rule off and nothing advancing on its own, so play
    only starts and stops from the dashboard. Real mode starts in HALT; Ctrl+C sends
    stop commands to the robots before exiting.

Before running on real robots:
  - ssl-vision's field geometry (length and width) must equal FIELD below, or the
    runner stops at the first geometry packet with "Field geometry mismatch".
  - The radio dongle is `PORT` in `utama_core/config/settings.py` (/dev/ttyACM0).
  - Real robots are capped at REAL_PARAMS.MAX_VEL (1 m/s, `config/robot_params.py`).
"""

import argparse
import dataclasses

from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import (
    AutoAdvanceConfig,
    GameConfig,
    RefereeProfile,
    RulesConfig,
)
from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic

# ---------------------------------------------------------------------------
# Test-day configuration
# ---------------------------------------------------------------------------

MODE = "real"  # "real", "rsim" or "grsim"; --mode overrides it

# PLACEHOLDER: measure the real field. Every value is a half-size in metres
# (half length, half width, ...), as in `STANDARD_FIELD_DIMS`.
FIELD = FieldDimensions(
    full_field_half_length=2.0,
    full_field_half_width=1.5,
    half_defense_area_depth=0.3,
    half_defense_area_width=0.6,
    half_goal_width=0.4,
    center_circle_radius=0.3,
)

# The part of the field the camera sees: "left" or "right" half (x < 0 or x > 0 in
# ssl-vision's coordinates), or "full". The goal in that half is the one in play.
CAMERA_HALF = "right"

ATTACKER_COLOUR = "yellow"  # the defender is the other colour

# Each robot's vision ID (its pattern on top) -> its firmware/radio ID. Swapping a
# robot is a change here only. The two firmware IDs must differ: one radio sends both.
ATTACKER_VISION_ID, ATTACKER_FIRMWARE_ID = 0, 0
DEFENDER_VISION_ID, DEFENDER_FIRMWARE_ID = 1, 1

# Robots whose ball sensor (IR) is trusted for `has_ball`, by vision ID. Put every
# robot with a working IR here. An untrusted one counts as having the ball once its
# centre is within 13 cm of the ball, whatever way it faces: the attacker can stop
# ~2 cm short and kick at nothing (seen in grSim), so watch for it if IR is off.
TRUSTED_IR_VISION_IDS: frozenset[int] = frozenset()

ATTACKER_TACTIC = "lead"  # see ATTACKER_TACTICS
DEFENDER_TACTIC = "goalkeeper"  # see DEFENDER_TACTICS

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


def _one_robot_strategy(robot_id: int, tactic_cls) -> AbstractStrategy:
    """`robot_id` plays `tactic_cls`, or keeps goal when `tactic_cls` is None."""
    outfield = () if tactic_cls is None else (robot_id,)

    def _build(motion_controller) -> KernelSchedulerStrategy:
        # A Strategy needs one registered tactic even with no outfield robot to run it.
        return KernelSchedulerStrategy(
            tactics={"play": (tactic_cls or DefenseTactic)()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "play"),
            outfield_robot_ids=outfield,
            ctx=TickContext(motion_controller=motion_controller),
        )

    return AbstractStrategy(
        build_kernel_strategy=_build,
        goalkeeper_id=robot_id if tactic_cls is None else _NO_ROBOT,
    )


def _operator_profile() -> RefereeProfile:
    """Every rule off and every transition left to the operator, except goals."""
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


def build_runner(mode: str) -> StrategyRunner:
    real = mode == "real"
    # In the simulators each team's robots are numbered from 0.
    attacker_id = ATTACKER_VISION_ID if real else 0
    defender_id = DEFENDER_VISION_ID if real else 0
    attacker_is_yellow = ATTACKER_COLOUR == "yellow"
    if ATTACKER_COLOUR not in ("yellow", "blue"):
        raise ValueError(f"ATTACKER_COLOUR must be 'yellow' or 'blue', got {ATTACKER_COLOUR!r}")

    # The defender's goal is the one in the camera's half, so the attacker's own goal
    # is on the other side. With the full field, the attacker attacks the right goal.
    attacker_is_right = CAMERA_HALF == "left"

    attacker_map = {ATTACKER_VISION_ID: ATTACKER_FIRMWARE_ID}
    defender_map = {DEFENDER_VISION_ID: DEFENDER_FIRMWARE_ID}
    attacker_ir = TRUSTED_IR_VISION_IDS & {ATTACKER_VISION_ID}
    defender_ir = TRUSTED_IR_VISION_IDS & {DEFENDER_VISION_ID}

    profile = _operator_profile()
    referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
    server = attach_dashboard(port=DASHBOARD_PORT)
    referee_view.attach(server, referee, profile)

    return StrategyRunner(
        strategy=_one_robot_strategy(attacker_id, ATTACKER_TACTICS[ATTACKER_TACTIC]),
        opp_strategy=_one_robot_strategy(defender_id, DEFENDER_TACTICS[DEFENDER_TACTIC]),
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
        yellow_trusted_ir_robots=(attacker_ir if attacker_is_yellow else defender_ir) if real else None,
        blue_trusted_ir_robots=(defender_ir if attacker_is_yellow else attacker_ir) if real else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default=MODE)
    args = parser.parse_args()
    build_runner(args.mode).run()


if __name__ == "__main__":
    main()
