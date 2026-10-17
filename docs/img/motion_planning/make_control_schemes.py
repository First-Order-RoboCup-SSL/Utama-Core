"""Draws control_schemes.png for docs/motion_planning_comparison.md: how move() reaches the three control
schemes (fpp, dwa, trajsample) through get_control_scheme, and what each one hands back.

pixi run python docs/img/motion_planning/make_control_schemes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "move",
    "get_control_scheme",
    "CONTROL_SCHEME_MAP",
    "MotionController",
    "FastPathPlanningController",
    "DWAController",
    "DWATranslationController",
    "DynamicWindowPlanner",
    "TrajectorySamplingController",
    "TrajectorySamplingPlanner",
    "TwoDPID",
    "get_pids",
    "_path_to",
    "RobotCommand",
    "StrategyRunner",
    "utama_core/skills/src/utils/move_utils.py",
    "utama_core/motion_planning/src/common/control_schemes.py",
    "utama_core/motion_planning/src/controllers/fastpathplanning.py",
    "utama_core/motion_planning/src/controllers/dwa_controller.py",
    "utama_core/motion_planning/src/controllers/trajsampling.py",
    "utama_core/motion_planning/src/pid/pid.py",
]

d = Diagram(
    180,
    104,
    "Control schemes: how move() reaches the planners",
    "move() calls the scheme's calculate once per robot per tick. pid is also registered but not drawn: it steers "
    "straight at the target, with no obstacle planning.",
    figsize=(14, 8.8),
)

# --- entry and registry ---------------------------------------------------------------------------
d.box(
    18,
    54,
    34,
    28,
    "move()",
    "skills/src/utils/move_utils.py\ncalls calculate(game, robot_id,\ntarget, orientation) once,\nthen rotates the velocity\ninto the robot frame",
    kind="decision",
)
d.box(
    56,
    54,
    34,
    28,
    "get_control_scheme",
    "common/control_schemes.py\nname -> class, in\nCONTROL_SCHEME_MAP\npicked per side by\nStrategyRunner (fpp)",
    kind="buffer",
)
d.arrow((35, 54), (39, 54))

# --- the three planners ---------------------------------------------------------------------------
d.box(
    108,
    82,
    56,
    22,
    "fpp: FastPathPlanningController",
    "controllers/fastpathplanning.py\nlookahead carrot from _path_to,\nTwoDPID steers to the carrot;\nthe braking cap uses the true target",
    kind="motion",
)
d.box(
    108,
    54,
    56,
    22,
    "dwa: DWAController",
    "controllers/dwa_controller.py\nDWATranslationController, through\nDynamicWindowPlanner, picks the velocity",
    kind="motion",
)
d.box(
    108,
    28,
    56,
    22,
    "trajsample: TrajectorySamplingController",
    "controllers/trajsampling.py\nTrajectorySamplingPlanner.plan: bang-bang\ncandidates, the first collision-free one;\n"
    "braked if the closing speed is too high",
    kind="motion",
    size=9.5,
)
d.arrow((73, 54), (80, 82))
d.arrow((73, 54), (80, 54))
d.arrow((73, 54), (80, 28))

# --- shared orientation ---------------------------------------------------------------------------
d.box(108, 6, 56, 10, "Orientation, all three", "pid/pid.py: get_pids, a PID to target_oren", kind="motion")

# --- output ---------------------------------------------------------------------------------------
d.box(
    160,
    52,
    34,
    28,
    "RobotCommand",
    "global velocity from calculate,\nrotated into the robot frame;\nangular velocity as it is",
    kind="output",
)
d.arrow((136, 82), (143, 62))
d.arrow((136, 54), (143, 52))
d.arrow((136, 28), (143, 42))
d.arrow((136, 6), (160, 38))

d.key(
    4,
    36,
    [
        ("decision", "skills: the caller of move()"),
        ("buffer", "the registry"),
        ("motion", "motion planning and control"),
        ("output", "where the command leaves"),
    ],
)
print(d.save(__file__, "control_schemes.png"))
