"""Draws real_robot_path.png for utama_core/team_controller/README.md ("Real robot path"): one tick of a real
match, from the SSL-Vision and referee multicast into the refiners, the Strategy and MotionController, and out to
the robots over the serial radio, with the feedback back into RobotInfoRefiner.

pixi run python docs/img/team_controller/make_real_robot_path.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "VisionReceiver",
    "RefereeMessageReceiver",
    "PositionRefiner",
    "CameraCombiner",
    "VelocityRefiner",
    "RobotInfoRefiner",
    "RefereeRefiner",
    "GameFrame",
    "Strategy",
    "MotionController",
    "RealRobotController",
    "RobotResponse",
    "VISION_PORT",
    "REFEREE_PORT",
    "PORT",
    "BAUD_RATE",
    "utama_core/config/settings.py",
    "utama_core/data_processing/receivers/vision_receiver.py",
    "utama_core/data_processing/receivers/referee_receiver.py",
    "utama_core/data_processing/refiners/position.py",
    "utama_core/data_processing/refiners/filters/kalman.py",
    "utama_core/data_processing/refiners/velocity.py",
    "utama_core/data_processing/refiners/robot_info.py",
    "utama_core/data_processing/refiners/referee.py",
    "utama_core/entities/game/game_frame.py",
    "utama_core/engine/strategy.py",
    "utama_core/motion_planning",
    "utama_core/team_controller/src/controllers/real/real_robot_controller.py",
]

d = Diagram(
    150,
    112,
    "Real robot path: one tick of a real match",
    "StrategyRunner wires it: receivers fill one-slot buffers, refiners build the GameFrame, and the real "
    "controller sends commands over the serial radio.",
    figsize=(14, 10.5),
)

VX, RX = 26, 124  # the vision column and the referee column

# --- network and receivers -----------------------------------------------------------------------
d.box(VX, 100, 46, 9, "SSL-Vision (multicast)", "224.5.23.2:10006 (VISION_PORT)", kind="source")
d.box(RX, 100, 46, 9, "Referee, GameController", "224.5.23.1:10003 (REFEREE_PORT)", kind="source")
d.receiver(VX, 88, 40, 7, "VisionReceiver", "data_processing/receivers/")
d.receiver(RX, 88, 40, 7, "RefereeMessageReceiver", "data_processing/receivers/")
d.arrow((VX, 95.5), (VX, 91.5))
d.arrow((RX, 95.5), (RX, 91.5))

# --- buffers -------------------------------------------------------------------------------------
d.box(VX, 76, 36, 6.5, "Vision buffers", "deque(maxlen=1) per camera", kind="buffer", size=10)
d.box(RX, 76, 36, 6.5, "Referee buffer", "deque(maxlen=1): only the latest", kind="buffer", size=10)
d.arrow((VX, 84.5), (VX, 79.3), "RawVisionData", lx=1.4)
d.arrow((RX, 84.5), (RX, 79.3), "RefereeData", lx=1.4)

# --- refiners: the main path runs along this row -------------------------------------------------
Y = 58
d.box(
    VX,
    Y,
    32,
    16,
    "PositionRefiner",
    "CameraCombiner merges the\ncameras; a Kalman filter\nper robot and ball",
    "refiner",
)
d.box(62, Y, 20, 16, "VelocityRefiner", "velocities and\naccelerations,\nfrom GameHistory", kind="refiner")
d.box(
    92,
    Y,
    22,
    16,
    "RobotInfoRefiner",
    "has-ball from the IR\nfeedback; from vision for\nuntrusted robots",
    kind="refiner",
)
d.box(RX, Y, 28, 16, "RefereeRefiner", "command, stage,\nscores, designated\nposition", kind="refiner")
d.arrow((VX, 72.8), (VX, Y + 8.4), "latest frame of each camera", lx=1.4)
d.arrow((RX, 72.8), (RX, Y + 8.4))
for a, b in ((42, 52), (72, 81), (103, RX - 14)):
    d.main((a, Y), (b, Y))

# --- game frame, strategy, motion ----------------------------------------------------------------
d.box(RX, 40, 30, 9, "GameFrame", "entities/game/game_frame.py", kind="state")
d.main((RX, Y - 8), (RX, 44.5))
d.box(RX, 24, 30, 12, "Strategy", "engine/strategy.py: the tactics\npick a target per robot", kind="decision")
d.main((RX, 35.5), (RX, 30.4))
d.box(82, 24, 26, 12, "MotionController", "motion_planning/: a target\nbecomes a velocity", kind="motion")
d.arrow((RX - 15, 24), (82 + 13, 24), "targets", lx=0, ly=2.4, ha="center")

# --- output and feedback -------------------------------------------------------------------------
d.box(
    30,
    24,
    40,
    12,
    "RealRobotController",
    "team_controller/src/controllers/real/\nserial radio on settings.PORT, 115200 baud",
    kind="output",
    size=10,
)
d.arrow((82 - 13, 24), (30 + 20, 24), "RobotCommand", lx=0, ly=2.4, ha="center")
d.arrow((30 + 10, 30.4), (92 - 8, Y - 8.4), ls="--")
d.note(60, 42, "RobotResponse (has ball)", ha="right", style="italic")
d.box(30, 7, 40, 7, "Robots", "over the serial radio", kind="source")
d.arrow((30, 18.4), (30, 10.9))

d.key(
    101,
    17,
    [
        ("source", "outside the process"),
        ("receiver", "receiver"),
        ("buffer", "one-slot buffer"),
        ("refiner", "refiner"),
    ],
)
d.key(
    126,
    17,
    [
        ("state", "game state"),
        ("decision", "strategy layer"),
        ("motion", "motion planning"),
        ("output", "robot controller"),
    ],
    title="",
)
print(d.save(__file__, "real_robot_path.png"))
