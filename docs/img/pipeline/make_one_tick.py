"""Draws one_tick.png for docs/pipeline_method.md and the README: one tick of StrategyRunner, from the data
sources to the robots, with the class or module behind each box.

pixi run python docs/img/pipeline/make_one_tick.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import LOOP, Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "StrategyRunner",
    "CONTROL_FREQUENCY",
    "TIMESTEP",
    "CustomReferee",
    "VisionReceiver",
    "RefereeMessageReceiver",
    "get_robots_responses",
    "RawVisionData",
    "RefereeData",
    "RobotResponse",
    "PositionRefiner",
    "CameraCombiner",
    "VelocityRefiner",
    "RobotInfoRefiner",
    "RefereeRefiner",
    "GameFrame",
    "GameHistory",
    "Game",
    "RefereeOverride",
    "MatchStats",
    "RobotCommand",
    "utama_core/run/strategy_runner.py",
    "utama_core/data_processing/receivers",
    "utama_core/entities/game",
    "utama_core/engine/strategy.py",
    "utama_core/motion_planning",
    "utama_core/team_controller/src/controllers",
]

d = Diagram(
    140,
    115,
    "One tick of StrategyRunner",
    "utama_core/run/strategy_runner.py, 60 times a second (CONTROL_FREQUENCY). In rsim both teams run in the "
    "same runner: each side refines and steps in turn.",
)

# --- sources -------------------------------------------------------------------------------------
d.box(24, 99, 36, 10, "Vision", "SSL Vision over UDP (real, grSim)\nor the rsim observation", kind="source")
d.box(70, 99, 36, 10, "Robots", "radio feedback (real)\nor the simulator's responses", kind="source")
d.box(116, 99, 36, 10, "Referee", "GameController over UDP, or\nCustomReferee stepped in process", kind="source")

# --- receivers -----------------------------------------------------------------------------------
d.receiver(24, 84, 34, 7, "VisionReceiver", "data_processing/receivers/, own thread")
d.receiver(70, 84, 34, 7, "get_robots_responses()", "team_controller/ RobotController")
d.receiver(116, 84, 34, 7, "RefereeMessageReceiver", "or CustomReferee.step (rsim)")
for x in (24, 70, 116):
    d.arrow((x, 93.6), (x, 88))

# --- buffers -------------------------------------------------------------------------------------
d.box(24, 71, 34, 6.5, "Vision buffers", "deque(maxlen=1) per camera", kind="buffer", size=10)
d.box(116, 71, 34, 6.5, "Referee buffer", "deque(maxlen=1): only the latest", kind="buffer", size=10)
d.arrow((24, 80.4), (24, 74.6), "RawVisionData", lx=1.2)
d.arrow((116, 80.4), (116, 74.6), "RefereeData", lx=1.2)

# --- refiners on the main loop -------------------------------------------------------------------
Y = 50
d.box(
    24,
    Y,
    30,
    15,
    "PositionRefiner",
    "CameraCombiner merges the\ncameras, then a Kalman\nfilter per robot and ball",
    "refiner",
)
d.box(58, Y, 22, 15, "VelocityRefiner", "velocities and\naccelerations,\nfrom GameHistory", kind="refiner")
d.box(
    87,
    Y,
    24,
    15,
    "RobotInfoRefiner",
    "has-ball from our IR,\nor inferred from positions;\nthe enemy's in rsim",
    "refiner",
)
d.box(116, Y, 22, 15, "RefereeRefiner", "command, stage,\nscores, designated\nposition", kind="refiner")
d.arrow((24, 67.6), (24, 57.6), "latest frame of each camera", lx=1.2)
d.arrow((70, 80.4), (87, 57.6), "RobotResponse\n(IR has-ball)", lx=1.5)
d.arrow((116, 67.6), (116, 57.6), "RefereeData", lx=1.2)
for a, b in ((39.2, 46.8), (69.2, 74.8), (99.2, 104.8)):
    d.main((a, Y), (b, Y))
d.note(135.2, 40, "GameFrame", colour=LOOP, weight="bold", ha="left", va="center", rotation=90, fontsize=9)

# --- state ---------------------------------------------------------------------------------------
d.box(116, 30, 28, 11, "Game", "latest GameFrame + GameHistory\n(entities/game/), immutable", kind="state")
d.loop([(127.2, Y), (134, Y), (134, 30), (130.6, 30)])
d.box(116, 13, 26, 9, "Recorded", "replay writer, MatchStats,\ndashboard vision stream", kind="output", size=10)
d.arrow((116, 24.4), (116, 17.6), ls="--")

# --- decision ------------------------------------------------------------------------------------
d.box(
    77,
    30,
    38,
    16,
    "Strategy.step",
    "engine/strategy.py. Restart, STOP or timeout:\nRefereeOverride places the robots.\nOtherwise the partitioner "
    "splits\nthe robots between tactics (tactics/),\nwhich call skills (skills/)",
    kind="decision",
)
d.main((101.4, 30), (96.6, 30))
d.box(
    34,
    30,
    26,
    16,
    "MotionController",
    "motion_planning/: fpp (default),\ndwa or trajsample turn each\ntarget into a velocity,\navoiding robots and the\n"
    "ball's keep-out",
    kind="motion",
)
d.arrow((57.4, 30), (47.6, 30))
d.note(52.5, 27.6, "targets", style="italic", ha="center", va="top")

# --- output --------------------------------------------------------------------------------------
d.box(34, 13, 28, 8, "RobotController", "team_controller/src/controllers/", kind="output", size=10)
d.arrow((34, 21.8), (34, 17.2), "RobotCommand per robot", lx=1.2)
for x, w, name in ((12, 13, "rsim"), (27, 13, "grSim"), (54, 16, "Real robots")):
    d.box(x, 3.5, w, 4.5, name, kind="output", size=9.5)
d.arrow((26, 8.8), (12, 5.9))
d.arrow((30, 8.8), (27, 5.9))
d.arrow((42, 8.8), (54, 5.9))
d.loop([(19.8, 13), (1.5, 13), (1.5, Y), (8.8, Y)])
d.note(2.6, 31, "next tick:\noff rsim, sleep\nto TIMESTEP", colour=LOOP, ha="left", va="center")

d.key(
    75,
    17.6,
    [
        ("receiver", "receiver"),
        ("refiner", "refiner"),
        ("state", "game state"),
        ("decision", "strategy layer"),
        ("motion", "motion"),
    ],
)
print(d.save(__file__, "one_tick.png"))
