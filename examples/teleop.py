# ruff: noqa: E402
"""examples/teleop.py — drive either robot by keyboard from the dashboard's Live tab.

Run:
    pixi run python examples/teleop.py               # real robots
    pixi run python examples/teleop.py --mode rsim   # rehearse in the simulator
or pick "teleop" in `pixi run demos`' Demo panel, between demos, to place robots and the ball.

Open http://localhost:8080, pick the robot in the Teleop panel, tick "keys drive" and Force
Start. W/S forward and back, A/D sideways, Q/E turn, Shift faster, B kicks, F turns the dribbler
on and off; Space halts as everywhere on the page. A robot stops when its keys are let go, when
the page stops sending (closed, the network gone) and on Halt. No rules, no goals: nothing but
the operator stops play. Robots are `shooting_drill.py`'s (`ROBOT_A`, `ROBOT_B` here), each
kept inside the area in play.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Optional

from shooting_drill import (
    CAMERA_HALF,
    DASHBOARD_PORT,
    FIELD,
    KEEP_OUT,
    RADIO_0,
    RADIO_1,
    _field_bounds,
    _one_robot_strategy,
    _referee_profile,
    on_field,
)

from utama_core.custom_referee import CustomReferee
from utama_core.dashboard import DashboardServer, attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.dashboard.views import teleop as teleop_view
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.teleop import TeleopDrive, TeleopTactic

ROBOT_A = RADIO_0
ROBOT_B = RADIO_1


def build_runner(mode: str, server: Optional[DashboardServer] = None) -> StrategyRunner:
    real = mode == "real"
    robot_a, robot_b = on_field(ROBOT_A, ROBOT_B) if real else (ROBOT_A, ROBOT_B)
    a_is_yellow = robot_a.colour == "yellow"
    # In the simulators each team's robots are numbered from 0.
    a_id = robot_a.vision_id if real else 0
    b_id = robot_b.vision_id if real else 0
    a_name, b_name = f"{robot_a.colour} {a_id}", f"{robot_b.colour} {b_id}"
    drive = TeleopDrive([a_name, b_name])

    profile = _referee_profile(rules_on=False, goals=False, kickoff=robot_a.colour)
    referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
    server = server or attach_dashboard(port=DASHBOARD_PORT)  # `demos.py` passes its own
    referee_view.attach(server, referee, profile)
    teleop_view.attach(server, drive)

    a_map = {robot_a.vision_id: robot_a.firmware_id}
    b_map = {robot_b.vision_id: robot_b.firmware_id}
    a_profiles, b_profiles = {a_id: robot_a.profile}, {b_id: robot_b.profile}
    return StrategyRunner(
        strategy=_one_robot_strategy(a_id, lambda: TeleopTactic(drive, a_name)),
        opp_strategy=_one_robot_strategy(b_id, lambda: TeleopTactic(drive, b_name)),
        my_team_is_yellow=a_is_yellow,
        my_team_is_right=not a_is_yellow,
        mode=mode,
        exp_friendly=1,
        exp_enemy=1,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        keep_out=KEEP_OUT,
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
        show_live_status=not real,
        yellow_vision_to_cmd_mapping=(a_map if a_is_yellow else b_map) if real else None,
        blue_vision_to_cmd_mapping=(b_map if a_is_yellow else a_map) if real else None,
        yellow_robot_profiles=a_profiles if a_is_yellow else b_profiles,
        blue_robot_profiles=b_profiles if a_is_yellow else a_profiles,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    build_runner(parser.parse_args().mode).run()


if __name__ == "__main__":
    main()
