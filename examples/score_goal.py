# ruff: noqa: E402
"""examples/score_goal.py — one robot, no opponent: it takes the ball and scores.

Run:
    pixi run python examples/score_goal.py            # real robot
    pixi run python examples/score_goal.py --mode rsim
    pixi run python examples/score_goal.py --calibrate   # fixed moves from the dashboard;
    # pixi run calibrate (examples/calibrate_robot.py) does it from the terminal, with the ball
    # the dashboard is at http://localhost:8080: press Force Start to play, Halt to freeze

The field and the camera's view are `FIELD` and `CAMERA_HALF` in `shooting_drill.py`,
so the two examples always agree; the robot is set below. It attacks the goal at +x (the right
one in ssl-vision's coordinates) with `LeadAndSupportTactic`: alone, it goes to the ball, turns
to the open corner and shoots. It starts in HALT, and the field check gates Force Start as in
`shooting_drill.py`. Put the ball back and Force Start again for the next shot; Halt and
Run in the Live tab's Strategy panel restarts the tactic.

`--calibrate` measures the robot alone, as `shooting_drill.py --calibrate` does with two
(that one halts on the second robot it never sees): it plays fixed moves
(`tactics/scripted_moves.py`) instead of shooting. Put it near the middle of the camera's view
facing +x, with 0.8 m clear ahead and 0.6 m to its left, and a ball at rest in view; Force Start.
Then: pixi run python tools/robot_calibration.py replays/field/<run>_calibrate.log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from typing import Optional

from shooting_drill import (
    CAMERA_HALF,
    DASHBOARD_PORT,
    FIELD,
    RADIO_0,
    RECORD,
    TestRobot,
    _field_bounds,
    _one_robot_strategy,
    _referee_profile,
    on_field,
)

from utama_core.custom_referee import CustomReferee
from utama_core.dashboard import DashboardServer, attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.dashboard.views import strategy as strategy_view
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.scripted_moves import ScriptedMovesTactic

# ---------------------------------------------------------------------------
# Test-day configuration
# ---------------------------------------------------------------------------

SHOOTER = RADIO_0

# ---------------------------------------------------------------------------


def build_runner(
    mode: str, tactics=None, name: str = "score_goal", server: Optional[DashboardServer] = None
) -> StrategyRunner:
    """The robot plays `tactics[next(iter(tactics))]`; the Strategy panel switches among `tactics`
    ({name: tactic class}, default this example's shooter). `name` tags the recorded log."""
    tactics = tactics or {"shoot": LeadAndSupportTactic}
    real = mode == "real"
    (shooter,) = on_field(SHOOTER) if real else (SHOOTER,)
    shooter_id = shooter.vision_id if real else 0  # in the simulators robots are numbered from 0
    is_yellow = shooter.colour == "yellow"
    mapping = {shooter.vision_id: shooter.firmware_id} if real else None

    profile = _referee_profile(kickoff=shooter.colour)
    referee = CustomReferee(profile, n_robots_yellow=int(is_yellow), n_robots_blue=int(not is_yellow))
    server = server or attach_dashboard(port=DASHBOARD_PORT)  # `demos.py` passes its own
    referee_view.attach(server, referee, profile)

    game_log = f"replays/field/{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{name}.log" if RECORD and real else None
    if game_log:
        print(f"Recording to {game_log}")
    first = next(iter(tactics))
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=_one_robot_strategy(shooter_id, tactics[first]),
        my_team_is_yellow=is_yellow,
        my_team_is_right=True,  # own goal at -x: it attacks the goal at +x
        mode=mode,
        exp_friendly=1,
        exp_enemy=0,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
        show_live_status=not real,
        yellow_vision_to_cmd_mapping=mapping if is_yellow else None,
        blue_vision_to_cmd_mapping=None if is_yellow else mapping,
        yellow_robot_profiles={shooter_id: shooter.profile} if is_yellow else None,
        blue_robot_profiles=None if is_yellow else {shooter_id: shooter.profile},
    )
    strategy_view.attach(
        server,
        runner,
        choices={"my": {n: (lambda t=t: _one_robot_strategy(shooter_id, t)) for n, t in tactics.items()}},
        running={"my": first},
    )
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    parser.add_argument("--calibrate", action="store_true", help="measure the robot: fixed moves")
    args = parser.parse_args()
    if args.calibrate:
        build_runner(args.mode, {"calibrate": ScriptedMovesTactic}, name="calibrate").run()
    else:
        build_runner(args.mode).run()


if __name__ == "__main__":
    main()
